/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef FIXEDDOWN_H
#define FIXEDDOWN_H

#include "FSM/FSMState.h"

class State_FixedDown : public FSMState
{
public:
    State_FixedDown(CtrlComponents *ctrlComp);
    ~State_FixedDown() {}
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();

private:
    float _targetPos_3[12] = {-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                              -0.5, 1.36, -2.65, 0.5, 1.36, -2.65};

    float _startPos[12];
    float _duration_1 = 700;   
    float _percent_1 = 0;      

    float Kp = 60.0;
    float Kd = 5.0;
};

#endif // FIXEDDOWN_H