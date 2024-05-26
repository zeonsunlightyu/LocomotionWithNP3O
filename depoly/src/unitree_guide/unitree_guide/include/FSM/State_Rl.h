/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef RL_H
#define RL_H

#include "FSM/FSMState.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "common/LowPassFilter.h"
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
    float _startPos[12] = {0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                           0.0, 0.67, -1.3, 0.0, 0.67, -1.3};
    float _duration_1 = 500;   // steps
    float _duration_2 = 10; // B2
    float _percent_1 = 0;    //%
    float _percent_2 = 0;    //%

    float stand_kp[12];
    float stand_kd[12];

    float Kp = 30.0;
    float Kd = 0.75;
    float infer_dt = 0.02;

    //gamepad
    float smooth = 0.03;
    float dead_zone = 0.01;

    float rx = 0.;
    float ly = 0.;
    float lx = 0.;

    std::vector<float> action;
    std::vector<float> action_temp;
    std::vector<float> prev_action;

    torch::Tensor action_buf;
    torch::Tensor obs_buf;
    torch::Tensor last_action;

    at::string model_path;
    torch::jit::script::Module model;
    torch::DeviceType device;

    bool threadRunning;
    std::thread* infer_thread;
    std::mutex write_cmd_lock;

    //gravity
    LPFilter *_gxFilter, *_gyFilter, *_gzFilter;
    std::vector<LPFilter*> action_filters;

    Estimator *_est;

    // Rob State
    RotMat _B2G_RotMat, _G2B_RotMat;
    Vec3 gravity;
    Vec3 forward;
    // default values
    //int history_length = 5;
    int history_length = 10;
//    float  init_pos[12] = {0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
//                       0.0, 0.9, -1.8, 0.0, 0.9, -1.8};
    float  init_pos[12] = {-0.1,0.8,-1.5,0.1,0.8,-1.5,-0.1,1.0,-1.5,0.1,1.0,-1.5};
    float pos_scale = 1.0;
    float vel_scale = 0.05;
    float action_scale[12] = {0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};
    float action_delta_max = 1.0;
    float action_delta_min = -1.0;
    //float action_scale[12] = {0.0,0.25,0.25,0.0,0.25,0.25,0.0,0.25,0.25,0.0,0.25,0.25};
};

#endif // RL_H
