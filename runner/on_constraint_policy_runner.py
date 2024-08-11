import time
import os
from collections import deque
import statistics
import warnings

from torch.utils.tensorboard import SummaryWriter
import torch
from global_config import ROOT_DIR

from modules import ActorCriticRMA,ActorCriticRmaTrans,ActorCriticSF,ActorCriticBarlowTwins,ActorCriticStateTransformer,ActorCriticTransBarlowTwins,ActorCriticMixedBarlowTwins,ActorCriticRnnBarlowTwins,ActorCriticVqvae
from algorithm import NP3O
from envs.vec_env import VecEnv
from modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from utils.helpers import hard_phase_schedualer, partial_checkpoint_load
from copy import copy, deepcopy

class OnConstraintPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        # self.phase1_end = self.cfg["phase1_end"] 
 
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCriticRMA = actor_critic_class(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg)
        if self.cfg['resume']:
            model_dict = torch.load(os.path.join(ROOT_DIR, self.cfg['resume_path']))
            actor_critic.load_state_dict(model_dict['model_state_dict'])
        
        actor_critic.to(self.device)
        

        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    self.policy_cfg["scan_encoder_dims"][-1], 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None

        # Create algorithm
        self.alg_cfg['k_value'] = self.env.cost_k_values
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = alg_class(actor_critic, 
                                  depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device,
                                  **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
            [self.env.cfg.cost.num_costs],
            self.env.cost_d_values_tensor
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.act_shed,self.imi_shed,self.lag_shed = hard_phase_schedualer(max_iters=tot_iter,
        #             phase1_end=self.phase1_end)

        #imitation_mode
        if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
            self.alg.actor_critic.imitation_mode()
            
        for it in range(self.current_learning_iteration, tot_iter):
            # act_teacher_flag = self.act_shed[it]
            # imi_flag = self.imi_shed[it]
            # lag_flag = self.lag_shed[it]

            # self.alg.set_imi_flag(imi_flag)
            # self.alg.actor_critic.set_teacher_act(act_teacher_flag)
            # # self.env.randomize_lag_timesteps = lag_flag
            # # if self.env.randomize_lag_timesteps:
            # #     print("lag is on")
            # # else:
            # #     print("lag is off")
            # if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
            #     step_size = 1/int(tot_iter/2)
            #     imi_weight = max(0,1 - it * step_size)
            #     print("imi_weight:",imi_weight)
            #     self.alg.set_imi_weight(imi_weight)
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                   
                    actions = self.alg.act(obs, critic_obs, infos)
                    obs, privileged_obs, rewards,costs,dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs,rewards,costs,dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device),costs.to(self.device),dones.to(self.device)
                    self.alg.process_env_step(rewards,costs,dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                self.alg.compute_cost_returns(critic_obs)

            #update k value for better expolration
            k_value = self.alg.update_k_value(it)
            
            mean_value_loss,mean_cost_value_loss,mean_viol_loss,mean_surrogate_loss, mean_imitation_loss,obs_batch_min,obs_batch_max = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        #mean_std = self.alg.actor_critic.std.mean()
        mean_std = self.alg.actor_critic.get_std().mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        #mean_kl_loss,mean_recons_loss,mean_vel_recons_loss
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/cost_value_function', locs['mean_cost_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_viol_loss', locs['mean_viol_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_imitation_loss', locs['mean_imitation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        self.writer.add_scalar('Data/obs_max', locs['obs_batch_max'], locs['it'])
        self.writer.add_scalar('Data/obs_min', locs['obs_batch_min'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
