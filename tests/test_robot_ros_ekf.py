"""Opt-in real ROS-process test, isolated from every running simulator.

FYP_TEST_ROS=1 python3 -m pytest -q tests/test_robot_ros_ekf.py
after sourcing /opt/ros/jazzy/setup.bash. Generated rates are controlled test
inputs, not benchmark observations. No ACE response is fabricated.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.environ.get("FYP_TEST_ROS")!="1",reason="requires explicit isolated ROS process test")
def test_real_ekf_raw_encoder_and_gyro(tmp_path,monkeypatch):
    pytest.importorskip("rclpy")
    import rclpy
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState,Imu
    from nav_msgs.msg import Odometry
    # A private DDS domain prevents this test's clock/sensors reaching Gazebo.
    domain=100+os.getpid()%100
    monkeypatch.setenv("ROS_DOMAIN_ID",str(domain))
    root=Path(__file__).parents[1]
    output=tmp_path/"ros"
    output.mkdir()
    monkeypatch.setenv("ROS_LOG_DIR",str(tmp_path/"ros_logs"))
    commands=[
        ["/opt/ros/jazzy/lib/robot_localization/ekf_node","--ros-args","--params-file",
         str(root/"configs/robot_sim/ekf.yaml"),"-r","__node:=ekf_filter_node"],
        [sys.executable,"-m","research3d.robot_sim.ros_localization_node","--config",
         str(root/"configs/robot_sim/localization.json"),"--output",str(output),"--duration-wall-s","45"]]
    processes=[]
    logs=[]
    node=None
    poses=[]
    try:
        for i,command in enumerate(commands):
            log=(tmp_path/f"child_{i}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT))
        rclpy.init(args=[])
        node=Node("controlled_encoder_imu_test")
        clock_pub=node.create_publisher(Clock,"/clock",10)
        joint_pub=node.create_publisher(JointState,"/robot/joint_states",10)
        imu_pub=node.create_publisher(Imu,"/robot/imu",10)
        node.create_subscription(Odometry,"/odometry/filtered",poses.append,100)
        began=time.monotonic()
        index=0
        while time.monotonic()-began<30.:
            ns=1000000000+index*20000000
            stamp=Clock()
            stamp.clock.sec=ns//1000000000
            stamp.clock.nanosec=ns%1000000000
            clock_pub.publish(stamp)
            joint=JointState()
            joint.header.stamp=stamp.clock
            joint.name=["left_wheel_joint","right_wheel_joint"]
            joint.position=[index*.04,index*.04]
            joint.velocity=[2.,2.]
            joint_pub.publish(joint)
            imu=Imu()
            imu.header.stamp=stamp.clock
            imu.header.frame_id="base_link"
            imu.angular_velocity.z=0.
            imu_pub.publish(imu)
            rclpy.spin_once(node,timeout_sec=.005)
            time.sleep(.015)
            index+=1
        assert all(p.poll() is None for p in processes),[(tmp_path/f"child_{i}.log").read_text() for i in range(2)]
        assert len(poses)>700,f"only {len(poses)} filtered odometry messages"
        last=poses[-1]
        assert last.header.frame_id=="odom" and last.child_frame_id=="base_link"
        assert 4.<last.pose.pose.position.x<8.
        assert abs(last.pose.pose.position.y)<.01
        assert abs(last.twist.twist.linear.x-.2)<.02
        timestamps=[x.header.stamp.sec+x.header.stamp.nanosec/1e9 for x in poses]
        rate=(len(timestamps)-1)/(timestamps[-1]-timestamps[0])
        assert 27.<rate<33.,f"EKF was {rate:.3f} Hz"
        pose_cov=sum(last.pose.covariance[i] for i in [0,7,35])
        assert pose_cov<100.,f"local covariance diverged: {pose_cov}"
        assert last.twist.covariance[7]<.01,"lateral velocity must be constrained"
        trace=[json.loads(x) for x in (output/"localization.jsonl").read_text().splitlines()]
        assert any(x["local_valid"] for x in trace)
        assert trace[-1]["local_state"]=="TRACKING"
        assert all(not x["global_valid"] for x in trace)
        assert all(x["accepted_count"]==0 and not x["ace_enabled"] for x in trace)
        (output/"controlled_ros_verification.json").write_text(json.dumps({
            "type":"controlled_raw_sensor_ROS_integration_not_benchmark", "ROS_DOMAIN_ID":domain,
            "filtered_messages":len(poses),"rate_hz":rate,
            "final_x_m":last.pose.pose.position.x,"velocity_m_s":last.twist.twist.linear.x,
            "final_pose_covariance_trace":pose_cov,"final_lateral_velocity_variance":last.twist.covariance[7],
            "fake_ace_observations":0,"GT_to_estimator":False},indent=2)+"\n")
    finally:
        for child in reversed(processes):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=6.)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3.)
        for log in logs:log.close()
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
