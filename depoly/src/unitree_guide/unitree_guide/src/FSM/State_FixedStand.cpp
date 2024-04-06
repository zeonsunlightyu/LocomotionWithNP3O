/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include "FSM/State_FixedStand.h"

State_FixedStand::State_FixedStand(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::FIXEDSTAND, "fixed stand"){}

void State_FixedStand::enter(){
    for(int i=0; i<4; i++){
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
            _lowCmd->setSimStanceGain(i);
        }
        else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
            _lowCmd->setRealStanceGain(i);
        }
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }
    for(int i=0; i<12; i++){
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
        _startPos[i] = _lowState->motorState[i].q;
    }
    _ctrlComp->setAllStance();
}

void State_FixedStand::run(){
    _percent_1 += (float)1 / _duration_1;
    _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
    if (_percent_1 < 1)
    {
        for (int j = 0; j < 12; j++)
        {
            _lowCmd->motorCmd[j].q = (1 - _percent_1) * _startPos[j] + _percent_1 * _targetPos_1[j];
        }
    }
    else
    {
        _percent_2 += (float)1 / _duration_2;
        _percent_2 = _percent_2 > 1 ? 1 : _percent_2;

        for (int j = 0; j < 12; j++)
        {
            _lowCmd->motorCmd[j].q = (1 - _percent_2) * _targetPos_1[j] + _percent_2 * _targetPos_2[j];
        }
    }
}

void State_FixedStand::exit(){
    _percent_1 = 0;
    _percent_2 = 0;
}

FSMStateName State_FixedStand::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    if(_lowState->userCmd == UserCommand::L2_X){
        return FSMStateName::FIXEDDOWN;
    }
    if(_lowState->userCmd == UserCommand::L1_Y){
        return FSMStateName::RL;
    }
    else{
        return FSMStateName::FIXEDSTAND;
    }
}