import numpy as np
import os
from datetime import datetime
from configs.go2_constraint_cnn_phase1 import Go2ConstraintCnnRoughPhase1Cfg, Go2ConstraintCnnRoughPhase1CfgPPO
from configs.go2_constraint_him import Go2ConstraintHimRoughCfg, Go2ConstraintHimRoughCfgPPO
from configs.go2_constraint_rnn import Go2ConstraintRnnRoughCfg, Go2ConstraintRnnRoughCfgPPO
from configs.go2_constraint_transformer_phase2 import Go2ConstraintTransRoughPhase2Cfg, Go2ConstraintTransRoughPhase2CfgPPO

import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from configs import Go2ConstraintRoughCfg,Go2ConstraintRoughCfgPPO,Go2ConstraintTransRoughPhase1Cfg,Go2ConstraintTransRoughPhase1CfgPPO,Go2ConstraintCnnRoughPhase2Cfg,Go2ConstraintCnnRoughPhase2CfgPPO
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("go2N3po",LeggedRobot,Go2ConstraintRoughCfg(),Go2ConstraintRoughCfgPPO())
    task_registry.register("go2N3poTransPhase1",LeggedRobot,Go2ConstraintTransRoughPhase1Cfg(),Go2ConstraintTransRoughPhase1CfgPPO())
    task_registry.register("go2N3poTransPhase2",LeggedRobot,Go2ConstraintTransRoughPhase2Cfg(),Go2ConstraintTransRoughPhase2CfgPPO())
    task_registry.register("go2N3poCnnPhase1",LeggedRobot,Go2ConstraintCnnRoughPhase1Cfg(),Go2ConstraintCnnRoughPhase1CfgPPO())
    task_registry.register("go2N3poCnnPhase2",LeggedRobot,Go2ConstraintCnnRoughPhase2Cfg(),Go2ConstraintCnnRoughPhase2CfgPPO())
    task_registry.register("go2N3poRnn",LeggedRobot,Go2ConstraintRnnRoughCfg(),Go2ConstraintRnnRoughCfgPPO())
    task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
    args = get_args()
    train(args)
