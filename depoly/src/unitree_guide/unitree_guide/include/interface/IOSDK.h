/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef IOSDK_H
#define IOSDK_H

#include "interface/IOInterface.h"

#ifndef ROBOT_TYPE_Go2
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#else
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/common/time/time_tool.hpp"
#include "unitree/common/thread/thread.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
using namespace unitree::common;
using namespace unitree::robot;
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);
#endif

#ifdef COMPILE_WITH_MOVE_BASE
    #include <ros/ros.h>
    #include <ros/time.h>
    #include <sensor_msgs/JointState.h>
#endif  // COMPILE_WITH_MOVE_BASE


class IOSDK : public IOInterface{
public:
IOSDK();
~IOSDK(){}
void sendRecv(const LowlevelCmd *cmd, LowlevelState *state);

private:
#ifndef ROBOT_TYPE_Go2
    UNITREE_LEGGED_SDK::UDP _udp;
    UNITREE_LEGGED_SDK::Safety _safe;
    UNITREE_LEGGED_SDK::LowCmd _lowCmd{};
    UNITREE_LEGGED_SDK::LowState _lowState{};
#else
    uint8_t HEAD[2] = {0xFE, 0xEF};
    pthread_mutex_t lowlevelmutex;
    unitree_go::msg::dds_::LowCmd_ _lowCmd{};
    unitree_go::msg::dds_::LowState_ _lowState{};
    unitree::common::ThreadPtr lowCmdWriteThreadPtr;
    unitree::common::ThreadPtr highStateWriteThreadPtr;
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    void InitLowCmd_dds();
    void LowCmdwriteHandler(); 
    void LowStateMessageHandler(const void *);
    uint32_t crc32_core(uint32_t* ptr, uint32_t len);
#endif

#ifdef COMPILE_WITH_MOVE_BASE
    ros::NodeHandle _nh;
    ros::Publisher _pub;
    sensor_msgs::JointState _joint_state;
#endif  // COMPILE_WITH_MOVE_BASE
};

#endif  // IOSDK_H