#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h> 

#include "robot_interface.hpp"
#include "gamepad.hpp"
#include "cfg.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;
namespace unitree::common
{
    class BasicUserController
    {
    public:
        BasicUserController() {}

        virtual void LoadParam(fs::path &param_folder) = 0;

        virtual void Reset(RobotInterface &robot_interface, Gamepad &gamepad) = 0;

        virtual void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) = 0;

        virtual void Calculate() = 0;

        virtual std::vector<float> GetLog() = 0;

        float dt, kp, kd, action_scale,command_scale,pos_scale,vel_scale;
        std::array<float,12> init_pos;
        std::array<float,12> init_gym_pos;
        std::array<float, 12> jpos_des;
    };

    class ExampleUserController : public BasicUserController
    {
    public:
        ExampleUserController() {}

        void LoadParam(fs::path &param_folder)
        {
            // load param file
            std::ifstream cfg_file(param_folder / "params.json");
            std::cout << "Read params from: " << param_folder / "params.json" << std::endl;
            std::stringstream ss;
            ss << cfg_file.rdbuf();
            FromJsonString(ss.str(), cfg);

            // get data from json
            dt = cfg.dt;
            kp = cfg.kp;
            kd = cfg.kd;
            action_scale = cfg.action_scale;
            pos_scale = cfg.pos_scale;
            vel_scale = cfg.vel_scale;
            history_length = cfg.history_length;
            model_path = param_folder/"model.pt";

            for (int i = 0; i < 12; ++i)
            {
                init_pos.at(i) = cfg.init_pos.at(i);
            }

            for (int i = 0; i < 12; ++i)
            {
                init_gym_pos.at(i) = cfg.init_gym_pos.at(i);
            }

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

            // initialize record
            action_buf = torch::zeros({history_length,12},device);
            obs_buf = torch::zeros({history_length,34},device);

            init_pos_tensor = torch::zeros({1,12});
            for (int i = 0; i < 12; ++i)
            {
                init_pos_tensor[0][i] = init_gym_pos[i];
            }
            init_pos_tensor = init_pos_tensor.to(device);

            last_action_tensor = torch::zeros({1,12},device);
            last_contact_tensor = torch::ones({1,4},device);
        }

        void GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
        {
            // record command
//            cmd.at(0) = gamepad.ly * 2.0;
//            cmd.at(1) = 0;//-gamepad.lx;
//            cmd.at(2) = 0;//-gamepad.rx;
//
//            // record robot state
//            for (int i = 0; i < 12; ++i)
//            {
//                jpos_processed.at(i) = robot_interface.jpos.at(i);
//                jvel_processed.at(i) = robot_interface.jvel.at(i);
//            }
//            for (int i = 0; i < 3; ++i)
//            {
//                gravity.at(i) = robot_interface.projected_gravity.at(i);
//            }
//            for (int i = 0; i < 4; ++i)
//            {
//                contact.at(i) = robot_interface.contact.at(i);
//            }

            cmd.push_back(gamepad.ly * 2.0);
            cmd.push_back(0);
            cmd.push_back(0);

            for (int i = 0; i < 12; ++i)
            {
                jpos_processed.push_back(robot_interface.jpos.at(i));
                jvel_processed.push_back(robot_interface.jvel.at(i));
            }

            for (int i = 0; i < 3; ++i)
            {
                gravity.push_back(robot_interface.projected_gravity.at(i));
            }
            for (int i = 0; i < 4; ++i)
            {
                contact.push_back(robot_interface.contact.at(i));
            }

            // gravity,cmd,dof_pos,dof_vel to tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat32);

            torch::Tensor gravity_tensor = torch::from_blob(gravity.data(),{1,3},options).to(device);
            torch::Tensor cmd_tensor = torch::from_blob(cmd.data(),{1,3},options).to(device);
            torch::Tensor dof_pos_tensor = torch::from_blob(jpos_processed.data(),{1,12},options).to(device);
            torch::Tensor dof_vel_tensor = torch::from_blob(jvel_processed.data(),{1,12},options).to(device);
            torch::Tensor contact_tensor = torch::from_blob(contact.data(),{1,4},options).to(device);
            // scale and offset
            dof_pos_tensor = (dof_pos_tensor - init_pos_tensor)*pos_scale;
            dof_vel_tensor = dof_vel_tensor*vel_scale;
            cmd_tensor = cmd_tensor*command_scale;
            contact_tensor = contact_tensor - 0.5;

            // concat obs
            torch::Tensor obs = torch::cat({gravity_tensor,cmd_tensor,dof_pos_tensor,dof_vel_tensor,contact_tensor},1).to(device);
            // append obs to obs buffer
            obs_buf = obs_buf.index({Slice(1,None),Slice()});
            obs_buf = torch::cat({obs_buf,obs},0);
        }

        void Reset(RobotInterface &robot_interface, Gamepad &gamepad)
        {
            GetInput(robot_interface, gamepad);
            Calculate();
        }

        void Calculate()
        {
            auto obs_buf_batch = obs_buf.unsqueeze(0);
            auto action_buf_batch = action_buf.unsqueeze(0);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(obs_buf_batch);
            inputs.push_back(action_buf_batch);
            // Execute the model and turn its output into a tensor.
            torch::Tensor action_raw = model.forward(inputs).toTensor();
            // interpolate action
            action_tensor = last_action_tensor*0.2 + action_raw*0.8;
            // record action raw
            last_action_tensor = action_raw.clone();
            // append to action buffer
            action_buf = action_buf.index({Slice(1,None),Slice()});
            action_buf = torch::cat({action_buf,action_raw},0);
            // scale action
            action_tensor = action_tensor * action_scale;
            // action modified by default pos
            action_tensor = action_tensor + init_pos_tensor;
            // assign to control
            action_tensor = action_tensor.squeeze(0);
            // move to cpu
            action_tensor = action_tensor.to(torch::kCPU);

            auto action_getter = action_tensor.accessor<float,1>();
            for (int i = 0; i < 12; ++i)
              {
                 jpos_des.at(i) = std::clamp(action_getter[i],action_high+jpos_processed.at(i),action_low+jpos_processed.at(i));
                 //std::cout << std::clamp(action_getter[i],action_high,action_low) << std::endl;
                 //std::cout << jpos_processed.at(i) << std::endl;
              }
           }

        std::vector<float> GetLog()
        {
            // record input, output and other info into a vector
            std::vector<float> log;
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(cmd.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jvel_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_des.at(i));
            }
            
            return log;
        }

        // cfg
        ExampleCfg cfg;

        // state
        std::vector<float> cmd;
        std::vector<float> gravity;
        std::vector<float> jpos_processed;
        std::vector<float> jvel_processed;
        std::vector<float> action;
        std::vector<float> contact;

        //record
        torch::Tensor last_contact_tensor;
        torch::Tensor last_action_tensor;
        torch::Tensor init_pos_tensor;
        torch::Tensor action_tensor;

        //buffer
        torch::Tensor action_buf;
        torch::Tensor obs_buf;
        int history_length;

        //policy model
        at::string model_path;
        torch::jit::script::Module model;
        torch::DeviceType device;
        float action_high = 1.0;
        float action_low = -1.0;
    };
} // namespace unitree::common
