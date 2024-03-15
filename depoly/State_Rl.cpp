#include <iostream>
#include "FSM/State_Rl.h"
#include <cmath>
#define _USE_MATH_DEFINES
using namespace std;


State_Rl::State_Rl(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::RL, "rl"),
    _est(ctrlComp->estimator)
{

}

void State_Rl::enter()
{
    // load policy
    model_path = "model.pt";
    load_policy();

    // initialize record
    action_buf = torch::zeros({history_length,12},device);
    obs_buf = torch::zeros({history_length,30},device);
    last_action = torch::zeros({1,12},device);

    // initialize default values
    gravity(0,0) = 0.0;
    gravity(1,0) = 0.0;
    gravity(2,0) = -1.0;

    forward(0,0) = 1.0;
    forward(1,0) = 0.0;
    forward(2,0) = 0.0;

    for (int j = 0; j < 12; j++)
    {
        action_temp.push_back(_lowState->motorState[j].q);
        action.push_back(_lowState->motorState[j].q);
    }

    for (int i = 0; i < history_length; i++)
    {
        torch::Tensor obs_tensor = get_obs();
        // append obs to obs buffer
        obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);
    }

    for (int i = 0; i < 10; i++)
    {
        model_infer();
    }

    // initialize thread
    threadRunning = true;
    infer_thread = new std::thread(&State_Rl::infer,this);


    // smooth transition of kp and kd
    float stand_kp[12];
    float stand_kd[12];
    for (int j = 0; j < 12; j++)
    {
        stand_kd[j] = _lowCmd->motorCmd[j].Kd;
        stand_kp[j] = _lowCmd->motorCmd[j].Kp;
    }
    // for(int i = 0; i < _duration_1; i++) {
    //     _percent_1 += (float) 1 / _duration_1;
    //     _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
    //     if (_percent_1 < 1) {
    //         for (int j = 0; j < 12; j++) {
    //             _lowCmd->motorCmd[j].Kp = (1 - _percent_1) * stand_kp[j] + _percent_1 * Kp;
    //             _lowCmd->motorCmd[j].Kd = (1 - _percent_1) * stand_kd[j] + _percent_1 * Kd;
    //         }
    //     }
    // }
}

void State_Rl::run()
{   
    if (_percent_1 < 1)
    {
        _percent_1 += (float) 1 / _duration_1;
        _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
        if (_percent_1 < 1) {
            for (int j = 0; j < 12; j++) {
                _lowCmd->motorCmd[j].Kp = (1 - _percent_1) * stand_kp[j] + _percent_1 * Kp;
                _lowCmd->motorCmd[j].Kd = (1 - _percent_1) * stand_kd[j] + _percent_1 * Kd;
            }
        } 
    }
    else
    {
        write_cmd_lock.lock();
        for (int j = 0; j < 12; j++)
        {
            _lowCmd->motorCmd[j].q = action.at(j);
            _lowCmd->motorCmd[j].dq = 0;
            _lowCmd->motorCmd[j].Kp = Kp;
            _lowCmd->motorCmd[j].Kd = Kd;
            _lowCmd->motorCmd[j].tau = 0;
        }
        write_cmd_lock.unlock();
    }
   
}

void State_Rl::exit()
{
    _percent_1 = 0;
    _percent_2 = 0;
    threadRunning = false;
    infer_thread->join();
}

torch::Tensor State_Rl::get_obs()
{
    std::vector<float> obs;
    // compute gravity
    _B2G_RotMat = _lowState->getRotMat();
    _G2B_RotMat = _B2G_RotMat.transpose();
    Vec3 projected_gravity = _G2B_RotMat*gravity;
    Vec3 projected_forward = _G2B_RotMat*forward;
    // gravity
    for (int i = 0; i < 3; ++i)
    {
        obs.push_back(projected_gravity(i,0));
    }

    // cmd
    float x_dir_vel = _lowState->userValue.lx * 1.5;
    float y_dir_vel = _lowState->userValue.ly * 1.0;
    float y_dir_rot = _lowState->userValue.rx * 3.14;

    obs.push_back(y_dir_vel);
    obs.push_back(x_dir_vel);

    double heading = atan2((double)forward(1,0), (double)forward(0,0));
    double angle = (double)y_dir_rot - heading;
    angle = fmod(angle,2.0*M_PI);
    if(angle > M_PI)
    {
        angle = angle - 2.0*M_PI;
    }
    angle = angle*0.5;

    float max = 1;
    float min = -1;
    angle = std::max(std::min((float)angle, max), min);
    angle = angle * 0.25;

    obs.push_back(angle);

//    obs.push_back(0.0);

    // pos
    for (int i = 0; i < 12; ++i)
    {
        float pos = (_lowState->motorState[i].q  - init_pos[i])* pos_scale;
        obs.push_back(pos);
    }
    // vel
    for (int i = 0; i < 12; ++i)
    {
        float vel = _lowState->motorState[i].dq * vel_scale;
        obs.push_back(vel);
    }

    // gravity,cmd,dof_pos,dof_vel to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,30},options).to(device);

    return obs_tensor;
}

torch::Tensor State_Rl::model_infer()
{
    auto obs_buf_batch = obs_buf.unsqueeze(0);
    auto action_buf_batch = action_buf.unsqueeze(0);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs_buf_batch);
    inputs.push_back(action_buf_batch);

    // Execute the model and turn its output into a tensor.
    torch::Tensor action_tensor = model.forward(inputs).toTensor();
    return action_tensor;
}

void State_Rl::infer()
{
    while(threadRunning)
    {
        long long _start_time = getSystemTime();

        torch::Tensor obs_tensor = get_obs();
        // append obs to obs buffer
        obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);

//        auto obs_buf_batch = obs_buf.unsqueeze(0);
//        auto action_buf_batch = action_buf.unsqueeze(0);
//
//        std::vector<torch::jit::IValue> inputs;
//        inputs.push_back(obs_buf_batch);
//        inputs.push_back(action_buf_batch);
//
//        // Execute the model and turn its output into a tensor.
//        torch::Tensor action_raw = model.forward(inputs).toTensor();
        torch::Tensor action_raw = model_infer();
        // action filter
        action_raw = 0.8*action_raw + 0.2*last_action;
        last_action = action_raw.clone();
        // append to action buffer
        action_buf = torch::cat({action_buf.index({Slice(1,None),Slice()}),action_raw},0);
        // assign to control
        action_raw = action_raw.squeeze(0);
        // move to cpu
        action_raw = action_raw.to(torch::kCPU);
        // assess the result
        auto action_getter = action_raw.accessor<float,1>();
        // should it be move up to start?
        //long long _start_time = getSystemTime();

        write_cmd_lock.lock();
        for (int j = 0; j < 12; j++)
        {
            action.at(j) = action_getter[j] * action_scale[j] + init_pos[j];
        }
        write_cmd_lock.unlock();

        absoluteWait(_start_time, (long long)(infer_dt * 1000000));
    }
    threadRunning = false;

}

void State_Rl::load_policy()
{
    // load model from check point
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device= torch::kCPU;
    if (torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    std::cout << "load model to device!" << std::endl;
    model.eval();
}

FSMStateName State_Rl::checkChange()
{
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    if(_lowState->userCmd == UserCommand::L2_X){
        return FSMStateName::FIXEDSTAND;
    }
    else{
        return FSMStateName::RL;
    }
}