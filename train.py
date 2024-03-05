import numpy as np
import os
from datetime import datetime

import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from configs import Go2ConstraintRoughCfg,Go2ConstraintRoughCfgPPO,Go2ConstraintTransRoughCfg,Go2ConstraintTransRoughCfgPPO
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("go2N3po",LeggedRobot,Go2ConstraintRoughCfg(),Go2ConstraintRoughCfgPPO())
    task_registry.register("go2N3poTrans",LeggedRobot,Go2ConstraintTransRoughCfg(),Go2ConstraintTransRoughCfgPPO())
    args = get_args()
    train(args)
