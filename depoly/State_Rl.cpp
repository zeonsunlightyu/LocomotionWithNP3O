#include <iostream>
#include "FSM/State_Rl.h"

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

    // initialize thread
    threadRunning = true;
    infer_thread = new std::thread(&State_Rl::infer,this);

    // initialize default values
    gravity(0,0) = 0.0;
    gravity(1,0) = 0.0;
    gravity(2,0) = -1.0;
}

void State_Rl::run()
{
    write_cmd_lock.lock();
    std::memcpy(action_temp.data(),action.data(),action.size());
    write_cmd_lock.unlock();

    for (int j = 0; j < 12; j++)
    {
        _lowCmd->motorCmd[j].q = action_temp.at(j);
        _lowCmd->motorCmd[j].dq = 0;
        _lowCmd->motorCmd[j].Kp = Kp;
        _lowCmd->motorCmd[j].Kd = Kd;
        _lowCmd->motorCmd[j].tau = 0;
    }
}

void State_Rl::exit()
{
    _percent_1 = 0;
    _percent_2 = 0;
    threadRunning = false;
    infer_thread->join();
}

void State_Rl::infer()
{
    while(threadRunning)
    {
        long long _start_time = getSystemTime();

        // compute gravity
        _B2G_RotMat = _lowState->getRotMat();
        _G2B_RotMat = _B2G_RotMat.transpose();
        Vec3 projected_gravity = _G2B_RotMat*gravity;

        // gravity
        for (int i = 0; i < 3; ++i)
        {
            obs.push_back(projected_gravity(i,0));
        }

        // cmd
        obs.push_back(1.0);
        obs.push_back(0);
        obs.push_back(0);

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

        // append obs to obs buffer
        obs_buf = obs_buf.index({Slice(1,None),Slice()});
        obs_buf = torch::cat({obs_buf,obs_tensor},0);

        auto obs_buf_batch = obs_buf.unsqueeze(0);
        auto action_buf_batch = action_buf.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs_buf_batch);
        inputs.push_back(action_buf_batch);

        // Execute the model and turn its output into a tensor.
        torch::Tensor action_raw = model.forward(inputs).toTensor();
        // append to action buffer
        action_buf = action_buf.index({Slice(1,None),Slice()});
        action_buf = torch::cat({action_buf,action_raw},0);
        // assign to control
        action_raw = action_raw.squeeze(0);
        // move to cpu
        action_raw = action_raw.to(torch::kCPU);
        // assess the result
        auto action_getter = action_raw.accessor<float,1>();

        write_cmd_lock.lock();
        for (int j = 0; j < 12; j++)
        {
            action.at(j) = action_getter[j] * action_scale + init_pos[j];
            // need to add hip scale
        }
        write_cmd_lock.unlock();

        absoluteWait(_start_time, (long long)(this->dt * 1000000));
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
        return FSMStateName::FIXEDDOWN;
    }
    else{
        return FSMStateName::FIXEDSTAND;
    }
}