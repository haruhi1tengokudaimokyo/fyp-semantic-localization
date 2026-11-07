"""Explicit ROS/Gazebo bridge allowlist. Offline ground truth is opt-in.

From repository root, after sourcing /opt/ros/jazzy/setup.bash:
  export ROS_DOMAIN_ID=71 GZ_PARTITION=fyp_robot_sim_20261107
  python3 -m research3d.robot_sim.sim_bridge

Add --offline-gt only for a separate recorder/evaluator. Ctrl-C stops this bridge
and its child, never other Gazebo or experiment processes. The bridge does not
start a simulator. To launch the generated world separately:
  gz sim -s -r --headless-rendering /absolute/path/airport_hall.sdf
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess

ONLINE_BRIDGES = (
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
    '/robot/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
    '/robot/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
    '/robot/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
    '/robot/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    '/robot/camera_right/image@sensor_msgs/msg/Image[gz.msgs.Image',
    '/robot/camera_right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    '/robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
)
OFFLINE_BRIDGES = ('/model/robot/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',)


def bridge_command(offline_gt=False):
    return ['ros2','run','ros_gz_bridge','parameter_bridge',*ONLINE_BRIDGES,
            *(OFFLINE_BRIDGES if offline_gt else ()),
            *(['--ros-args','-r','/model/robot/pose:=/robot/ground_truth'] if offline_gt else ())]


def main():
    p=argparse.ArgumentParser();p.add_argument('--offline-gt',action='store_true')
    args=p.parse_args()
    child=subprocess.Popen(bridge_command(args.offline_gt),start_new_session=True)
    previous_term=signal.getsignal(signal.SIGTERM)
    def terminate_request(_signal,_frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate_request)
    code=0
    try:
        code=child.wait()
    except KeyboardInterrupt:
        code=130
    finally:
        signal.signal(signal.SIGTERM,previous_term)
        # ros2 run creates a bridge grandchild. Stop only our new process group,
        # so direct signals to this parent cannot leave an orphan bridge alive.
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise SystemExit(code)


if __name__ == '__main__': main()
