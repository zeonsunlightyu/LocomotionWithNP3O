/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef RL_H
#define RL_H

#include "FSM/FSMState.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <thread>

using namespace torch::indexing;

class State_Rl : public FSMState
{
public:
    State_Rl(CtrlComponents *ctrlComp);
    ~State_Rl() {}
    void enter();
    void run();
    void exit();
    void infer();
    torch::Tensor get_obs();
    void load_policy();
    torch::Tensor model_infer();
    FSMStateName checkChange();

private:
    float _targetPos_1[12] = {0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                              0.0, 1.36, -2.65, 0.0, 1.36, -2.65};
    float _targetPos_2[12] = {0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
                              0.0, 0.9, -1.8, 0.0, 0.9, -1.8};
    float _startPos[12];
    float _duration_1 = 100;   // steps
    float _duration_2 = 500; // B2
    float _percent_1 = 0;    //%
    float _percent_2 = 0;    //%

    float Kp = 20.0;
    float Kd = 0.5;
    float infer_dt = 0.02;

    std::vector<float> action;
    std::vector<float> action_temp;

    torch::Tensor action_buf;
    torch::Tensor obs_buf;
    torch::Tensor last_action;

    at::string model_path;
    torch::jit::script::Module model;
    torch::DeviceType device;

    bool threadRunning;
    std::thread* infer_thread;
    std::mutex write_cmd_lock;

    Estimator *_est;

    // Rob State
    RotMat _B2G_RotMat, _G2B_RotMat;
    Vec3 gravity;
    Vec3 forward;
    // default values
    int history_length = 5;
    float  init_pos[12] = {0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
                       0.0, 0.9, -1.8, 0.0, 0.9, -1.8};
    float pos_scale = 1.0;
    float vel_scale = 0.05;
    float action_scale[12] = {0.0625,0.25,0.25,0.0625,0.25,0.25,0.0625,0.25,0.25,0.0625,0.25,0.25};
};

#endif // RL_H