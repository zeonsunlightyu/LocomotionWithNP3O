import cv2
import os

from isaacgym import gymapi
from envs import LeggedRobot
from modules import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from configs import *
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    #env_cfg.terrain.mesh_type = 'plane'
    env_cfg.domain_rand.push_robots = False
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    model_dict = torch.load(os.path.join(ROOT_DIR, 'model_3000.pt'))
    policy.load_state_dict(model_dict['model_state_dict'])
    policy = policy.to(env.device)

    # clear images under frames folder
    frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    delete_files_in_directory(frames_path)

    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 100
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    #torch.sum(self.last_actions - self.actions, dim=1)
    # self.base_lin_vel[:, 2]
    #torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0


    for i in range(num_frames):
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        env.commands[:,0] = 1
        env.commands[:,1] = 0
        env.commands[:,2] = 0
        env.commands[:,3] = 0
        actions = policy.act_inference(obs,hist_encoding=True)
        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)
        if RECORD_FRAMES:
            frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
            if not os.path.isdir(frames_path):
                os.mkdir(frames_path)
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512,512,4))[:,:,:3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 
    print("action rate:",action_rate/num_frames)
    print("z vel:",z_vel/num_frames)
    print("xy_vel:",xy_vel/num_frames)
    print("feet air reward",feet_air_time/num_frames)

    video.release()
if __name__ == '__main__':
    task_registry.register("go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
    task_registry.register("go1N3PO",LeggedRobot,Go1ConstraintRoughCfg(),Go1ConstraintRoughCfgPPO())
    RECORD_FRAMES = False
    args = get_args()
    play(args)
