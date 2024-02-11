import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F 
from modules.common_modules import BetaVAE, StateHistoryEncoder, get_activation, mlp_factory

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        actor_student_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        
        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)
        self.actor_student_backbone = nn.Sequential(*actor_student_layers)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean,_ = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop,latent], dim=1)
        mean = self.actor_student_backbone(backbone_input)
        return mean,latent
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        latent = self.infer_hist_latent(obs)
        backbone_input = torch.cat([obs_prop,latent], dim=1)
        mean = self.actor_teacher_backbone(backbone_input)
        return mean,latent
        
    def evaluate(self, obs, **kwargs):
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        scan_latent = self.scan_encoder(obs_scan)   
        obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        latent = self.infer_hist_latent(obs)
       
        backbone_input = torch.cat([obs_prop_scan,latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        scan_latent = self.scan_encoder(obs_scan)   
        obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        latent = self.infer_hist_latent(obs)
       
        backbone_input = torch.cat([obs_prop_scan,latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def imitation_learning_loss(self, obs):
        with torch.no_grad():
            target_mean,target_latent = self.act_teacher(obs)
        mean,latent = self.act_student(obs)

        loss = F.mse_loss(mean,target_mean) + F.mse_loss(latent,target_latent)
        return loss
