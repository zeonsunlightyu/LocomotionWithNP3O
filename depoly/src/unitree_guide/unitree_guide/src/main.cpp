/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include <unistd.h>
#include <csignal>
#include <sched.h>

#include "control/ControlFrame.h"
#include "control/CtrlComponents.h"
#include "Gait/WaveGenerator.h"
#include "control/BalanceCtrl.h"

#ifdef COMPILE_WITH_REAL_ROBOT
#include "interface/IOSDK.h"
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
using namespace unitree::robot::go2;
#endif // COMPILE_WITH_REAL_ROBOT

#ifdef COMPILE_WITH_ROS
#include "interface/KeyBoard.h"
#include "interface/IOROS.h"
#endif // COMPILE_WITH_ROS

bool running = true;

// over watch the ctrl+c command
void ShutDown(int sig)
{
    std::cout << "stop the controller" << std::endl;
    running = false;
}

void setProcessScheduler()
{
    pid_t pid = getpid();
    sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(pid, SCHED_FIFO, &param) == -1)
    {
        std::cout << "[ERROR] Function setProcessScheduler failed." << std::endl;
    }
}

//int queryServiceStatus(unitree::robot::go2::RobotStateClient rsc,const std::string& serviceName)
//{
//    std::vector<ServiceState> serviceStateList;
//    int ret,serviceStatus;
//    ret = rsc.ServiceList(serviceStateList);
//    size_t i, count=serviceStateList.size();
//    for (i=0; i<count; i++)
//    {
//        const ServiceState& serviceState = serviceStateList[i];
//        if(serviceState.name == serviceName)
//        {
//            if(serviceState.status == 0)
//            {
//                std::cout << "name: " << serviceState.name <<" is activate"<<std::endl;
//                serviceStatus = 1;
//            }
//            else
//            {
//                std::cout << "name:" << serviceState.name <<" is deactivate"<<std::endl;
//                serviceStatus = 0;
//            }
//        }
//    }
//    return serviceStatus;
//}
//
//void activateService(unitree::robot::go2::RobotStateClient rsc,const std::string& serviceName,int activate)
//{
//    rsc.ServiceSwitch(serviceName, activate);
//}


int main(int argc, char **argv)
{
    /* set real-time process */
    setProcessScheduler();
    /* set the print format */
    std::cout << std::fixed << std::setprecision(3);

#ifdef RUN_ROS
    ros::init(argc, argv, "unitree_gazebo_servo");
#endif // RUN_ROS

    IOInterface *ioInter;
    CtrlPlatform ctrlPlat;

#ifdef COMPILE_WITH_SIMULATION
    ioInter = new IOROS();
    ctrlPlat = CtrlPlatform::GAZEBO;
#endif // COMPILE_WITH_SIMULATION

#ifdef COMPILE_WITH_REAL_ROBOT
#ifdef ROBOT_TYPE_Go2

    unitree::robot::ChannelFactory::Instance()->Init(0,argv[1]);
    unitree::robot::go2::RobotStateClient rsc;
    rsc.SetTimeout(10.0f);
    rsc.Init();
    while(queryServiceStatus(rsc,"sport_mode"))
    {
        std::cout<<"Try to deactivate the service: "<<"sport_mode"<<std::endl;
        activateService(rsc,"sport_mode",0);
        sleep(1);
    }

#endif
    ioInter = new IOSDK();

    ctrlPlat = CtrlPlatform::REALROBOT;
#endif // COMPILE_WITH_REAL_ROBOT

    CtrlComponents *ctrlComp = new CtrlComponents(ioInter);
    ctrlComp->ctrlPlatform = ctrlPlat;
    ctrlComp->dt = 0.002; // run at 500hz
    ctrlComp->running = &running;

#ifdef ROBOT_TYPE_A1
    ctrlComp->robotModel = new A1Robot();
#endif
#ifdef ROBOT_TYPE_Go1
    ctrlComp->robotModel = new Go1Robot();
#endif
#ifdef ROBOT_TYPE_Go2
    ctrlComp->robotModel = new Go2Robot();
#endif
    ctrlComp->waveGen = new WaveGenerator(0.35, 0.5, Vec4(0, 0.5, 0.5, 0)); // Trot
    // ctrlComp->waveGen = new WaveGenerator(1.1, 0.75, Vec4(0, 0.25, 0.5, 0.75));  //Crawl, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.6, Vec4(0, 0.5, 0.5, 0));  //Walking Trot, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.35, Vec4(0, 0.5, 0.5, 0));  //Running Trot, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.7, Vec4(0, 0, 0, 0));  //Pronk, only for sim

    ctrlComp->geneObj();

    ControlFrame ctrlFrame(ctrlComp);

    signal(SIGINT, ShutDown);

    while (running)
    {
        ctrlFrame.run();
    }

    delete ctrlComp;
    return 0;
}
