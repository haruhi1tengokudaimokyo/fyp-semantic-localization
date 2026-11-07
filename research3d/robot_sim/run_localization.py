"""Bounded launcher for official ROS EKF and our GT-free ACE sensor bridge.

Requires an already running Gazebo/ROS bridge. Does not start or stop Gazebo,
other experiments, GPUs or system services. Every process terminated is an
explicit child PID launched here. No global pkill or unbounded monitoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--duration-wall-s",type=float,default=120.)
    p.add_argument("--domain-id",type=int,default=71)
    p.add_argument("--config",type=Path,default=Path("configs/robot_sim/localization.json"))
    p.add_argument("--ekf-config",type=Path,default=Path("configs/robot_sim/ekf.yaml"))
    p.add_argument("--ace-command-json")
    p.add_argument("--spvloc-command-json")
    p.add_argument("--allow-spvloc-corrections",action="store_true",
                   help="Experimental ablation: allow original SPVLoc recovery gate; default performs shadow inference only")
    args=p.parse_args(argv)
    if args.duration_wall_s<=0 or not 0<=args.domain_id<=232:
        p.error("positive bounded duration and valid ROS domain required")
    import rclpy  # fail early if ROS setup has not been sourced
    manifest=json.loads(args.map_manifest.read_text())
    config=json.loads(args.config.read_text())
    config["global"]["map_id"]=manifest["map_id"]
    config["global"]["map_sha256"]=manifest["map_sha256"]
    config["global"]["allow_spvloc_corrections"]=args.allow_spvloc_corrections
    config["T_base_camera_optical"]=manifest["T_base_camera_optical"]
    config["wheel"]["radius_m"]=manifest["sensors"]["wheel_radius_m"]
    config["wheel"]["separation_m"]=manifest["sensors"]["wheel_separation_m"]
    args.output.mkdir(parents=True,exist_ok=False)
    config_path=(args.output/"localization_config.json").resolve()
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    env=os.environ.copy()
    env["ROS_DOMAIN_ID"]=str(args.domain_id)
    env["ROS_LOG_DIR"]=str((args.output/"ros_logs").resolve())
    env.setdefault("GZ_PARTITION","fyp_robot_sim_20261107")
    commands=[
        ["/opt/ros/jazzy/lib/robot_localization/ekf_node","--ros-args","--params-file",str(args.ekf_config.resolve()),
         "-r","__node:=ekf_filter_node"],
        [sys.executable,"-m","research3d.robot_sim.ros_localization_node","--config",str(config_path),
         "--output",str(args.output.resolve()),"--duration-wall-s",str(args.duration_wall_s)]]
    if args.ace_command_json:
        commands[1]+=["--ace-command-json",args.ace_command_json]
    if args.spvloc_command_json:
        commands[1]+=["--spvloc-command-json",args.spvloc_command_json]
    if args.allow_spvloc_corrections:
        commands[1]+=["--allow-spvloc-corrections"]
    processes=[]
    logs=[]
    interrupted=False
    started=time.monotonic()
    def stop_request(signum,frame):
        nonlocal interrupted
        interrupted=True
    old_handlers={s:signal.signal(s,stop_request) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for name,command in zip(["ekf","global_node"],commands):
            log=(args.output/(name+".log")).open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env))
        (args.output/"launcher.json").write_text(json.dumps({"child_pids":[x.pid for x in processes],
            "commands":commands,"ROS_DOMAIN_ID":args.domain_id,"duration_wall_s":args.duration_wall_s,
            "map_sha256":manifest["map_sha256"],"gt_topics_subscribed":[],
            "spvloc_corrections_allowed":args.allow_spvloc_corrections,
            "mode":"wheel_imu"+("_ace" if args.ace_command_json else "")+("_spvloc" if args.spvloc_command_json else "")},indent=2)+"\n")
        while not interrupted and time.monotonic()-started<args.duration_wall_s+3.:
            codes=[x.poll() for x in processes]
            if any(c is not None for c in codes):
                if codes[1]==0 and time.monotonic()-started>=args.duration_wall_s:
                    break
                raise RuntimeError(f"localization child exited prematurely: {codes}")
            time.sleep(.2)
    finally:
        # Stop the subscriber/worker first, then the local EKF.
        for child in reversed(processes):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=8.)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=3.)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=3.)
        for log in logs:log.close()
        for s,handler in old_handlers.items():signal.signal(s,handler)
        (args.output/"launcher_exit.json").write_text(json.dumps({"elapsed_wall_s":time.monotonic()-started,
            "exit_codes":[x.returncode for x in processes],"interrupted":interrupted,
            "terminated_only_own_child_pids":True},indent=2)+"\n")


if __name__=="__main__":main()
