import numpy as np
import os
from datetime import datetime

import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from configs import Go1RoughCfg,Go1RoughCfgPPO,Go1ConstraintRoughCfg,Go1ConstraintRoughCfgPPO
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
    task_registry.register("go1N3PO",LeggedRobot,Go1ConstraintRoughCfg(),Go1ConstraintRoughCfgPPO())
    args = get_args()
    train(args)
