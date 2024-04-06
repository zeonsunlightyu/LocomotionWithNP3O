/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include "FSM/State_FixedDown.h"

State_FixedDown::State_FixedDown(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::FIXEDDOWN, "fixed down")
{

}

void State_FixedDown::enter()
{
    for (int i = 0; i < 4; i++)
    {
        if (_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO)
        {
            _lowCmd->setSimStanceGain(i);
        }
        else if (_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT)
        {
            _lowCmd->setRealStanceGain(i);
        }
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }

    for (int i = 0; i < 12; i++)
    {
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
        _startPos[i] = _lowState->motorState[i].q;
    }
    _ctrlComp->setAllStance();
}

void State_FixedDown::run()
{

    _percent_1 += (float)1 / _duration_1;
    _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
    if (_percent_1 < 1)
    {
        for (int j = 0; j < 12; j++)
        {
            _lowCmd->motorCmd[j].q = (1 - _percent_1) * _startPos[j] + _percent_1 * _targetPos_3[j];
            _lowCmd->motorCmd[j].dq = 0;
            _lowCmd->motorCmd[j].Kp = 60;
            _lowCmd->motorCmd[j].Kd = 5;
            _lowCmd->motorCmd[j].tau = 0;
        }
    
    }
}

void State_FixedDown::exit()
{
    _percent_1 = 0;    
}

FSMStateName State_FixedDown::checkChange()
{
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    
    if(_percent_1>=1)
        return FSMStateName::PASSIVE;
    else
        return FSMStateName::FIXEDDOWN;
}