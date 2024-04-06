Packages Version: v3.8.0

# Introduction
This package can send control command to real robot from ROS. You can do low-level control(namely control all joints on robot) and high-level control(namely control the walking direction and speed of robot).

This version is suitable for unitree_legged_sdk v3.5.1, namely Go1 robot.

## Packages:

Basic message function: `unitree_legged_msgs`

The interface between ROS and real robot: `unitree_legged_real`

## Environment
We recommand users to run this package in Ubuntu 18.04 and ROS melodic environment

## Dependencies
* [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk/releases)
### Notice
The newest release [v3.8.0](https://github.com/unitreerobotics/unitree_legged_sdk/releases/tag/3.8.0) only supports for robot: Go1.

Check release [v3.3.4](https://github.com/unitreerobotics/unitree_legged_sdk/releases/tag/3.3.4) for A1 support.

# Configuration
Before compiling this package, please download the corresponding unitree_legged_sdk as noted above, and put it to your own workspace's source folder(e.g. `~/catkin_ws/src`). Be careful with the sdk folder name. It should be "unitree_legged_sdk" without version tag.

# Build
You can use catkin_make to build ROS packages. First copy the package folder to `~/catkin_ws/src`, then:
```
cd ~/catkin_ws
catkin_make
```

# Setup the net connection
First, please connect the network cable between your PC and robot. Then run `ifconfig` in a terminal, you will find your port name. For example, `enx000ec6612921`.

Then, open the `ipconfig.sh` file under the folder `unitree_legged_real`, modify the port name to your own. And run the following commands:
```
sudo chmod +x ipconfig.sh
sudo ./ipconfig.sh
```
If you run the `ifconfig` again, you will find that port has `inet` and `netmask` now.
In order to set your port automatically, you can modify `interfaces`:
```
sudo gedit /etc/network/interfaces
```
And add the following 4 lines at the end:
```
auto enx000ec6612921
iface enx000ec6612921 inet static
address 192.168.123.162
netmask 255.255.255.0
```
Where the port name have to be changed to your own.

# Run the package
You can control your real Go1 robot from ROS by this package.

Before you run expamle program, please run command

```
roslaunch unitree_legged_real real.launch ctrl_level:=highlevel
```
or
```
roslaunch unitree_legged_real real.launch ctrl_level:=lowlevel
```

It depends which control mode you want to use.

Then, if you want to run high-level control mode, you can run example_walk node like this
```
rosrun unitree_legged_real example_walk
```

If you want to run low-level control mode, you can run example_position program node like this
```
rosrun unitree_legged_real example_postion
```

You can also run the node state_sub to subscribe the feedback information from Go1 robot
```
rosrun unitree_legged_real state_sub
```

You can also run the launch file that enables you control robot via keyboard like you can do in turtlesim package
```
roslaunch unitree_legged_real keyboard_control.launch
```

And before you do the low-level control, please press L2+A to sit the robot down and then press L1+L2+start to make the robot into
mode in which you can do joint-level control, finally make sure you hang the robot up before you run low-level control.
