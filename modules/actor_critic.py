import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F 
from modules.common_modules import AutoEncoder, BetaVAE, MixedMlp, RnnBarlowTwinsStateHistoryEncoder, RnnStateHistoryEncoder, StateHistoryEncoder, get_activation, mlp_factory, mlp_layernorm_factory
from modules.transformer_modules import ActionCausalTransformer, StateCausalClsTransformer, StateCausalHeadlessTransformer, StateCausalTransformer
class Config:
    def __init__(self):
        self.n_obs = 45
        self.block_size = 9
        self.n_action = 45+3
        self.n_layer: int = 4
        self.n_head: int = 4
        self.n_embd: int = 32
        self.dropout: float = 0.0
        self.bias: bool = True

class CnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 priv_encoder_output_dim,
                 actor_hidden_dims=[256, 256, 256],
                 activation='elu'):
        super(CnnActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.priv_encoder_output_dim = priv_encoder_output_dim
        self.activation = activation
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        self.actor_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        self.actor = nn.Sequential(*self.actor_layers)
    
    def forward(self,obs,hist):
        latent = self.history_encoder(hist)
        backbone_input = torch.cat([obs,latent], dim=1)
        mean = self.actor(backbone_input)
        return mean
    
class RnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 encoder_dims,
                 decoder_dims,
                 actor_dims,
                 encoder_output_dim,
                 hidden_dim,
                 num_actions,
                 activation,) -> None:
        super(RnnActor,self).__init__()
        self.rnn_encoder = RnnStateHistoryEncoder(activation_fn=activation,
                                                  input_size=num_prop,
                                                  encoder_dims=encoder_dims,
                                                  hidden_size=hidden_dim,
                                                  output_size=encoder_output_dim)
        self.next_state_decoder =nn.Sequential(*mlp_factory(activation=activation,
                                              input_dims=hidden_dim,
                                              out_dims=num_prop+7,
                                              hidden_dims=decoder_dims))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=hidden_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        latents = self.rnn_encoder(obs_hist_full)
        actor_input = torch.cat([latents[:,-1,:],obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    def predict_next_state(self,obs_hist):
        # self.rnn_encoder.reset_hidden()
        latents = self.rnn_encoder(obs_hist)
        predicted = self.next_state_decoder(latents[:,-1,:])
        return predicted
    
class RnnBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 obs_encoder_dims,
                 rnn_encoder_dims,
                 actor_dims,
                 encoder_output_dim,
                 hidden_dim,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(RnnBarlowTwinsActor,self).__init__()
        self.rnn_encoder = RnnBarlowTwinsStateHistoryEncoder(activation_fn=activation,
                                                  input_size=num_prop,
                                                  hidden_size=hidden_dim,
                                                  output_size=encoder_output_dim,
                                                  final_output_size=latent_dim,
                                                  encoder_dims=rnn_encoder_dims)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        latents = self.rnn_encoder(obs_hist_full)
        actor_input = torch.cat([latents,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,weight):
        b = obs.size()[0]
        hist_latent = self.rnn_encoder(obs_hist)
        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)
        # c = torch.sum(c,dim=0,keepdim=True)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + weight*off_diag
        return loss
    
<<<<<<< Updated upstream
=======
class RnnBarlowTwinsLipActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 rnn_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(RnnBarlowTwinsLipActor,self).__init__()
        self.rnn_encoder = RnnBarlowTwinsStateHistoryEncoder(activation_fn=activation, 
                                                             input_size=num_prop, 
                                                             encoder_dims=rnn_encoder_dims,
                                                             hidden_size=64,
                                                             output_size=64,
                                                             final_output_size=latent_dim+7)

        self.actor = MixedLipMlp(input_size=num_prop,
                              latent_size=latent_dim+7,
                              hidden_size=64,
                              num_actions=num_actions,
                              num_experts=8)

        
        self.obs_encoder = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        latents = self.rnn_encoder(obs_hist_full[:,5:,:])
        #actor_input = torch.cat([latents,obs],dim=-1)
        # mean  = self.actor(actor_input)
        mean  = self.actor(latents,obs)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        predicted = self.rnn_encoder(obs_hist[:,5:,:])
        hist_latent = predicted[:,7:]
        priv_latent = predicted[:,:7]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)

        lip_gate_loss = self.actor.get_gate_lip_loss()
        loss = on_diag + weight*off_diag + 0.01*priv_loss + lip_gate_loss
        return loss
    
>>>>>>> Stashed changes
class MlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBarlowTwinsActor,self).__init__()
        self.mlp_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                 out_dims=latent_dim+7,
                                 hidden_dims=mlp_encoder_dims))

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 7,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        latents = self.mlp_encoder(obs_hist_full.view(b,-1))
        actor_input = torch.cat([latents,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    # def BarlowTwinsLoss(self,obs,obs_hist,weight):
    #     b = obs.size()[0]
    #     obs_hist = obs_hist[:,5:,:].view(b,-1)
    #     hist_latent = self.mlp_encoder(obs_hist)
    #     obs_latent = self.obs_encoder(obs)

    #     c = self.bn(hist_latent).T @ self.bn(obs_latent)
    #     c.div_(b)

    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()
    #     loss = on_diag + weight*off_diag
    #     return loss
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        predicted = self.mlp_encoder(obs_hist.view(b,-1))
        hist_latent = predicted[:,7:]
        priv_latent = predicted[:,:7]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss
<<<<<<< Updated upstream
=======

    
class MlpBVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBVAEActor,self).__init__()


        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop +3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = BetaVAE(in_dim=num_hist*num_prop,beta=0.1)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist_full[:,5:,:].reshape(b,-1))
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs,obs_hist,priv):
        obs = obs.detach()
        obs_hist = obs_hist.detach()
        
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,z, mu, log_var,predicted_vel  = self.Vae(obs_hist[:,5:,:].reshape(b,-1))
        loss = self.Vae.loss_fn(obs,recon,mu,log_var)
        mseloss = F.mse_loss(predicted_vel,priv)

        loss = loss + mseloss
        return loss

class MlpVQVAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpVQVAEActor,self).__init__()


        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop +3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.Vae = VQVAE_vel(in_dim=num_hist*num_prop)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latents,predicted_vel = self.Vae.get_latent(obs_hist_full[:,5:,:].reshape(b,-1))
            # predicted_vel = self.history_encoder(obs_hist_full)

        # normed = self.norm(torch.cat([predicted_vel.detach(),obs.detach()],dim=-1))
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs.detach()],dim=-1)
        # mean  = self.actor(torch.cat([latents.detach(),predicted_vel.detach()],dim=-1),obs.detach())
        # actor_input = torch.cat([normed,obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def VaeLoss(self,obs,obs_hist,priv):
        obs = obs.detach()
        obs_hist = obs_hist.detach()
        
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].view(b,-1)
        # recon,quantize,z,onehot_encode = self.Vae(obs_hist[:,5:,:].reshape(b,-1))
        # loss = self.Vae.loss_fn(obs,recon,quantize,z,onehot_encode) 
        recon,quantize,z,predicted_vel = self.Vae(obs_hist[:,5:,:].reshape(b,-1))
        # predicted_vel = self.history_encoder(obs_hist)
        loss = self.Vae.loss_fn(obs,recon,quantize,z) 
        mseloss = F.mse_loss(predicted_vel,priv)
        loss = loss + mseloss

        return loss

class MlpMAEActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpMAEActor,self).__init__()


        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=num_prop + 16,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
       
        self.mae = MAE(in_dim=num_hist*num_prop,latent_dim=16)
        

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latents = self.mae.get_latent(obs_hist_full[:,5:,:].view(b,-1))
        actor_input = torch.cat([latents.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def maeLoss(self,obs,obs_hist,priv):
        obs = obs.detach()
        obs_hist = obs_hist.detach()
        
        b = obs.size()[0]

        obs_hist = obs_hist[:,5:,:].view(b,-1)
        recon,recon_est  = self.mae(obs_hist)
        loss = self.mae.loss_fn(obs_hist,recon,torch.cat([obs,priv],dim=-1),recon_est) 

        return loss
    
class MlpSimSiamActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSimSiamActor,self).__init__()

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims= num_prop + 16 + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
    
        self.encoder = nn.Sequential(   nn.Linear(num_hist*num_prop, 128),
                                        nn.ELU(inplace=True),
                                        nn.Linear(128, 64),
                                        nn.ELU(inplace=True),
                                        nn.Linear(64, 16),
                                        nn.LayerNorm(16, elementwise_affine=False)) # output layer
        
        self.scan_encoder = nn.Sequential(nn.Linear(187, 128),
                                        nn.ELU(inplace=True),
                                        nn.Linear(128, 64),
                                        nn.ELU(inplace=True),
                                        nn.Linear(64, 16),
                                        nn.LayerNorm(16, elementwise_affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(16, 8),
                                        nn.LayerNorm(8),
                                        nn.ELU(inplace=True), # hidden layer
                                        nn.Linear(8, 16)) # output layer

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist*2, 3,final_act=False)

        self.criterion = nn.CosineSimilarity(dim=1)
        
    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latents = self.encoder(obs_hist_full[:,5:,:].reshape(b,-1))
            # latents,predicted_vel = self.encoder(obs_hist_full[:,5:,:])
            predicted_vel = self.history_encoder(obs_hist_full)

        # actor_input = torch.cat([latents.detach(),obs.detach()],dim=-1)
        actor_input = torch.cat([latents.detach(),predicted_vel.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        return mean
    
    def SimSiamLoss(self,obs,obs_hist,priv,scan):

        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        
        obs = obs.detach()
        # obs_hist_full = obs_hist_full.detach()
        obs_hist = obs_hist.detach()
        
        b = obs.size()[0]
        predicted_vel = self.history_encoder(obs_hist)
        priv_loss = F.mse_loss(predicted_vel,priv)

        z1 = self.encoder(obs_hist_full[:,5:,:].reshape(b,-1))
        z2 = self.scan_encoder(scan)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        z1 = z1.detach()
        z2 = z2.detach()

        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5 + 1.0
        loss = loss + priv_loss
        return loss
    
>>>>>>> Stashed changes
    
class MixedMlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MixedMlpBarlowTwinsActor,self).__init__()
        self.mlp_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                 out_dims=latent_dim+7,
                                 hidden_dims=mlp_encoder_dims))

        self.actor = MixedMlp(input_size=num_prop,
                              latent_size=latent_dim+7,
                              hidden_size=128,
                              num_actions=num_actions,
                              num_experts=4)
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        latents = self.mlp_encoder(obs_hist_full)
        #actor_input = torch.cat([latents,obs],dim=-1)
        mean  = self.actor(latents,obs)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        obs_hist = obs_hist[:,5:,:].view(b,-1)
        predicted = self.mlp_encoder(obs_hist)
        hist_latent = predicted[:,7:]
        priv_latent = predicted[:,:7]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss
    
class TransMlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(TransMlpBarlowTwinsActor,self).__init__()

        self.transformer_config = Config()
        self.transformer_config.n_layer = 4
        self.transformer_config.n_action = latent_dim + 7
        
        self.trans_encoder = StateCausalClsTransformer(self.transformer_config)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 7,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        obs_hist_full = obs_hist_full[:,5:,:]
        latents = self.trans_encoder(obs_hist_full)
        mean  = self.actor(torch.cat([latents,obs],dim=-1))
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        obs_hist = obs_hist[:,5:,:]
        predicted = self.trans_encoder(obs_hist)
        hist_latent = predicted[:,7:]
        priv_latent = predicted[:,:7]

        obs_latent = self.obs_encoder(obs)

        c = self.bn(hist_latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(priv_latent,priv)
        loss = on_diag + weight*off_diag + 0.01*priv_loss
        return loss

class TransBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 obs_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(TransBarlowTwinsActor,self).__init__()
        self.transformer_config = Config()
        self.transformer_config.n_layer = 2
        
        self.trans_encoder = StateCausalHeadlessTransformer(self.transformer_config)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=32,
                                 out_dims=num_actions,
                                 hidden_dims=[64]))
        
        self.obs_projector = nn.Sequential(*mlp_factory(activation=activation,
                            input_dims=32,
                            out_dims=latent_dim,
                            hidden_dims=[64]))
        
        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=activation,
                                 input_dims=num_prop,
                                 out_dims=latent_dim,
                                 hidden_dims=obs_encoder_dims))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
        ], dim=1)
        latent = self.trans_encoder(obs_hist_full)
        mean  = self.actor(latent)
        return mean
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        latent = self.trans_encoder(obs_hist)
        latent = self.obs_projector(latent)

        obs_latent = self.obs_encoder(obs)

        c = self.bn(latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + weight*off_diag
        return loss

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

class AeActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 encoder_dims,
                 decoder_dims,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(AeActor,self).__init__()
        self.ae = AutoEncoder(activation_fn=activation,
                            input_size=num_prop*num_hist,
                            encoder_dims=encoder_dims,
                            decoder_dims=decoder_dims,
                            latent_dim=latent_dim,
                            output_size=num_prop)
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

    def forward(self,obs,obs_hist):
        # self.rnn_encoder.reset_hidden()
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        b,t,n = obs_hist_full.size()
        obs_hist_full = obs_hist_full.view(b,-1)
        latent = self.ae.encode(obs_hist_full)
        actor_input = torch.cat([latent,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    def predict_next_state(self,obs_hist):
        b,t,n = obs_hist.size()
        obs_hist_flatten = obs_hist.view(b,-1)
        latent = self.ae.encode(obs_hist_flatten)
        predicted = self.ae.decode(latent)
        return predicted,latent
<<<<<<< Updated upstream
        
class StateCausalTransformerActor(nn.Module):
    def __init__(self):
        super(StateCausalTransformerActor,self).__init__()
        self.transformer_config = Config()
        self.transformer_config.n_layer = 3
        self.transformer_config.n_action = 12
        self.transformer = StateCausalClsTransformer(self.transformer_config)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        
        action = self.transformer(obs_hist_full[:,5:,:])
        return action
    
    def predict_next_action(self,obs_hist):
        predicted_action = self.transformer(obs_hist[:,5:,:])
        return predicted_action
    
class StateCausalTransformerBarlowTwinsActor(nn.Module):
    def __init__(self):
        super(StateCausalTransformerBarlowTwinsActor,self).__init__()
        self.transformer_config = Config()
        self.transformer_config.n_embd = 32
        self.transformer_config.n_layer = 2
        latent_dim = 16

        self.transformer = StateCausalHeadlessTransformer(self.transformer_config)

        self.action_head = nn.Sequential(
            nn.Linear(self.transformer_config.n_embd, self.transformer_config.n_embd),
            nn.GELU(),
            nn.Linear(self.transformer_config.n_embd, 12)
        )

        self.obs_head = nn.Sequential(
            nn.Linear(self.transformer_config.n_embd, self.transformer_config.n_embd),
            nn.GELU(),
            nn.Linear(self.transformer_config.n_embd, latent_dim)
        )

        self.obs_encoder = nn.Sequential(*mlp_layernorm_factory(activation=get_activation('elu'),
                                 input_dims=self.transformer_config.n_obs,
                                 out_dims=latent_dim,
                                 hidden_dims=[512,256,128]))
        
        self.bn = nn.BatchNorm1d(latent_dim,affine=False)

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        
        latent = self.transformer(obs_hist_full[:,5:,:])
        action = self.action_head(latent)
        return action
    
    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        b = obs.size()[0]
        latent = self.transformer(obs_hist[:,5:,:])
        latent = self.obs_head(latent)

        obs_latent = self.obs_encoder(obs)

        c = self.bn(latent).T @ self.bn(obs_latent)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + weight*off_diag 
        return loss
    
    def predict_next_action(self,obs_hist):
        #predicted_state = self.transformer(obs_hist[:,5:,:])
        latent= self.transformer(obs_hist[:,5:,:])
        predicted_action = self.action_head(latent)
        return predicted_action
    
class StateCausalTransformerVelActor(nn.Module):
    def __init__(self):
        super(StateCausalTransformerVelActor,self).__init__()
        self.transformer_config = Config()
        self.transformer_config.n_layer = 3

        self.transformer = StateCausalHeadlessTransformer(self.transformer_config)

        self.action_head = nn.Sequential(
            nn.Linear(self.transformer_config.n_embd+6, self.transformer_config.n_embd),
            nn.GELU(),
            nn.Linear(self.transformer_config.n_embd, 12)
        )

        self.vel_head = nn.Sequential(
            nn.Linear(self.transformer_config.n_embd, self.transformer_config.n_embd),
            nn.GELU(),
            nn.Linear(self.transformer_config.n_embd, 3)
        )

    def forward(self,obs,obs_hist):
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        
        latent = self.transformer(obs_hist_full[:,5:,:])
        vel = self.vel_head(latent)
        action_input = torch.cat([latent,vel,obs[:,6:9]],dim=-1)
        action = self.action_head(action_input)
        return action
    
    def predict_next_state(self,obs_hist,cmd):
        latent= self.transformer(obs_hist[:,5:,:])
        vel = self.vel_head(latent)
        action_input = torch.cat([latent,vel,cmd],dim=-1)
        action = self.action_head(action_input)

        return action,vel
    
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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with student actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)
        # actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+32,num_actions,actor_hidden_dims,last_act=False)

        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)
        self.actor_student_backbone = CnnActor(num_prop=num_prop,
                                               num_hist=num_hist,
                                               num_actions=num_actions,
                                               priv_encoder_output_dim=priv_encoder_output_dim,
                                               actor_hidden_dims=actor_hidden_dims,
                                               activation=activation)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        if self.teacher_act:
            mean = self.act_teacher(obs)
        else:
            mean = self.act_student(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        hist = obs[:, -self.num_hist*self.num_prop:].view(-1,self.num_hist,self.num_prop)
        mean = self.actor_student_backbone(obs_prop,hist)
        return mean
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]

        # scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,hist_latent], dim=1)
        mean = self.actor_teacher_backbone(backbone_input)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
     
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        with torch.no_grad():
            target_mean = self.act_teacher(obs)
        mean = self.act_student(obs)

        loss = F.mse_loss(mean,target_mean.detach())
        return loss
    
    def imitation_mode(self):
        self.actor_teacher_backbone.eval()
        self.scan_encoder.eval()
        self.priv_encoder.eval()
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
        model_jit = torch.jit.trace(self.actor_student_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

    
# class Config:
#     def __init__(self):
#         self.n_obs = 45
#         self.block_size = 9
#         self.n_action = 12
#         self.n_layer: int = 4
#         self.n_head: int = 4
#         self.n_embd: int = 32
#         self.dropout: float = 0.0
#         self.bias: bool = True

class ActorCriticRmaTrans(nn.Module):
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
        super(ActorCriticRmaTrans, self).__init__()

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
        
        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with student actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")
        
        
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        # self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        # actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        
        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)
        self.config = Config()
        self.actor_student_backbone = ActionCausalTransformer(self.config)

        # Value function
        # critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,last_act=False)
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        # cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+priv_encoder_output_dim,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        if self.teacher_act:
            mean,_ = self.act_teacher(obs)
        else:
            mean,_ = self.act_student(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, obs, **kwargs):
        obs_history_start = self.num_prop + self.num_priv_latent + self.num_scan + self.num_hist * self.num_prop
        obs_history_end = obs_history_start + self.num_hist*(self.num_prop-self.num_actions)
        action_history_start = obs_history_end

        obs_history = obs[:,obs_history_start:obs_history_end].view(-1,self.num_hist,self.num_prop-self.num_actions)
        action_history = obs[:,action_history_start:].view(-1,self.num_hist,self.num_actions)

        mean = self.actor_student_backbone(obs_history,action_history)
        return mean,None
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]

        scan_latent = self.infer_scandots_latent(obs)
        # hist_latent = self.infer_hist_latent(obs)
        latent = self.infer_priv_latent(obs)

        # backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)

        mean = self.actor_teacher_backbone(backbone_input)
        return mean,latent
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        # hist_latent = self.infer_hist_latent(obs)
        latent = self.infer_priv_latent(obs)

        # backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        # hist_latent = self.infer_hist_latent(obs)
        latent = self.infer_priv_latent(obs)

        # backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    # def infer_hist_latent(self, obs):
    #     hist = obs[:, -self.num_hist*self.num_prop:]
    #     return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        with torch.no_grad():
            target_mean,_ = self.act_teacher(obs)
        mean,_ = self.act_student(obs)

        loss = F.mse_loss(mean,target_mean.detach())
        return loss
    
    def imitation_mode(self):
        self.actor_teacher_backbone.eval()
        self.scan_encoder.eval()
        self.priv_encoder.eval()
    
    def save_torch_jit_policy(self,path,device):
        action_demo_input = torch.rand(1,5,12).to(device)
        obs_demo_input = torch.rand(1,5,self.config.n_obs).to(device)
        model_jit = torch.jit.trace(self.actor_student_backbone,(obs_demo_input,action_demo_input))
        model_jit.save(path)


class ActorCriticSF(nn.Module):
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
        super(ActorCriticSF, self).__init__()

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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        self.actor_teacher_backbone = RnnActor(num_prop=num_prop,
                                      num_actions=num_actions,
                                      encoder_dims=[128],
                                      decoder_dims=[128],
                                      actor_dims=[512,256,128],
                                      encoder_output_dim=32,
                                      hidden_dim=128,
                                      activation=activation)
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 7]
        target = torch.cat([priv,obs_prop],dim=-1)

        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        predicted = self.actor_teacher_backbone.predict_next_state(obs_hist)
        loss = F.mse_loss(predicted,target)
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)
=======
>>>>>>> Stashed changes

class ActorCriticBarlowTwins(nn.Module):
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
        super(ActorCriticBarlowTwins, self).__init__()

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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        # self.actor_teacher_backbone = RnnBarlowTwinsActor(num_prop=num_prop,
        #                               num_actions=num_actions,
        #                               actor_dims=[512,256,128],
        #                               encoder_output_dim=32,
        #                               hidden_dim=128,
        #                               activation=activation,
        #                               latent_dim=64,
        #                               obs_encoder_dims=[256,128],
        #                               rnn_encoder_dims=[128])
        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpBarlowTwinsActor(num_prop=num_prop,
                                      num_hist=10,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[256,128])
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 7]

        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).half().to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).half().to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

<<<<<<< Updated upstream
class ActorCriticMixedBarlowTwins(nn.Module):
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
        super(ActorCriticMixedBarlowTwins, self).__init__()

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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 16)

        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MixedMlpBarlowTwinsActor(num_prop=num_prop,
                                      num_hist=5,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[256,128])
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+16,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+16,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 7]

        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).half().to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).half().to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

class ActorCriticStateTransformer(nn.Module):
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
        super(ActorCriticStateTransformer, self).__init__()

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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with student actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 16)

        #state transformer
        self.actor_student_backbone = StateCausalTransformerActor()
        print(self.actor_student_backbone)
        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)

        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+16,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+16,cost_dims,critic_hidden_dims,last_act=False)
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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

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
        if self.teacher_act:
            mean = self.act_teacher(obs)
        else:
            mean = self.act_student(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]

        latent = self.infer_priv_latent(obs)
        # hist_latent = self.infer_hist_latent(obs)
        scan_latent = self.infer_scandots_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        mean = self.actor_teacher_backbone(backbone_input)
        return mean
    
    def act_student(self,obs,**kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_student_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        # target_action = obs_prop[:,-12:]
        # priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 3]

        # predicted_action,predicted_vel = self.actor_student_backbone.predict_next_state(obs_hist,obs_prop[:,6:9])

        # loss = 0.1*F.mse_loss(predicted_action,target_action) + 0.1*F.mse_loss(predicted_vel,priv)
        with torch.no_grad():
            target_action = self.act_teacher(obs)
        predicted_action = self.act_student(obs)

        loss = 0.1*F.mse_loss(predicted_action,target_action.detach())
        return loss
    
    def imitation_mode(self):
        print('turning actor teacher into eval')
        self.actor_teacher_backbone.eval()
        self.scan_encoder.eval()
        self.priv_encoder.eval()
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).half().to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).half().to(device)

        self.actor_student_backbone.eval()
        model_jit = torch.jit.trace(self.actor_student_backbone,(obs_demo_input,hist_demo_input))
        print(model_jit.code)
        model_jit.save(path)

class ActorCriticTransBarlowTwins(nn.Module):
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
        super(ActorCriticTransBarlowTwins, self).__init__()

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

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)

        # TransBarlowTwinsActor
        self.actor_teacher_backbone = TransMlpBarlowTwinsActor(num_prop=num_prop,
                                      num_hist=5,
                                      num_actions=num_actions,
                                      actor_dims=[256,128],
                                      mlp_encoder_dims=[512,256,128],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[256,128])
        print(self.actor_teacher_backbone)

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
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
        # return torch.clamp(self.std,0,1)
    
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
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 7]

        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)
=======
>>>>>>> Stashed changes
