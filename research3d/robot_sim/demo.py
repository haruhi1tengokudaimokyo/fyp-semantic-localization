"""Bounded complete Gazebo/ROS demo launcher; owns only its child processes.

Source ROS Jazzy before invoking. A map-specific trained ACE binding is required
for absolute localization. This starts the actual simulator, bridge and EKF,
then drives the predeclared physical route and records independent offline GT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-manifest",type=Path,required=True)
    p.add_argument("--ace-binding",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--route",type=Path,default=Path("configs/robot_sim/query_route.json"))
    p.add_argument("--seconds",type=float,default=60.)
    p.add_argument("--domain-id",type=int,default=72)
    p.add_argument("--spvloc-command-json")
    p.add_argument("--spvloc-references",type=Path)
    p.add_argument("--mapping-dataset",type=Path)
    p.add_argument("--spvloc-checkpoint",type=Path,default=Path("/data/fyp_3d/runs/semantic_localization_20261103/spvloc_100_20_20_float_targets_v2/checkpoints/epochepoch=18-stepstep=6517.ckpt"))
    p.add_argument("--allow-spvloc-corrections",action="store_true",help="experimental ablation only: permit currently unqualified SPVLoc candidates through original gates")
    p.add_argument("--events",type=Path,help="independent simulator disturbance protocol")
    p.add_argument("--camera-blackout",type=float,nargs=2,metavar=("START_S","END_S"),help="declared image-interface camera fault, both stereo views")
    a=p.parse_args()
    if a.spvloc_references and (a.spvloc_command_json or not a.mapping_dataset):p.error("SPV references require mapping dataset, and cannot combine with explicit RPC argv")
    if not 1<=a.seconds<=600 or not 1<=a.domain_id<=232:p.error("bounded duration and isolated valid ROS domain required")
    manifest=json.loads(a.world_manifest.read_text())
    binding=json.loads(a.ace_binding.read_text())
    if (binding["map_id"],binding["map_sha256"])!=(manifest["map_id"],manifest["map_sha256"]):
        raise ValueError("ACE belongs to a different map/world")
    if hashlib.sha256(Path(manifest["world_path"]).read_bytes()).hexdigest()!=manifest["map_sha256"]:
        raise ValueError("actual Gazebo SDF changed since map/model binding")
    import rclpy  # require the installed ROS environment, never install over it
    a.output.mkdir(parents=True,exist_ok=False)
    root=a.output.resolve();run=root/"localization"
    env=os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(a.domain_id),GZ_PARTITION=f"fyp_demo_{os.getpid()}",ROS_LOG_DIR=str(root/"ros_logs"))
    children=[];child_names=[];logs=[];interrupted=False
    began=time.monotonic();route_began=None;route_completed=None
    def interrupt(signum,frame):
        nonlocal interrupted
        interrupted=True
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,interrupt)
    def launch(name,argv):
        handle=(root/(name+".log")).open("w");logs.append(handle)
        child=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        children.append(child)
        child_names.append(name)
        return child
    try:
        launch("gazebo",["gz","sim","-s","-r","--headless-rendering",manifest["world_path"]])
        launch("bridge",[sys.executable,"-m","research3d.robot_sim.sim_bridge","--offline-gt"])
        # Waiting for the simulator service is bounded and checks actual replies.
        ready=False
        for _ in range(12):
            if interrupted:raise KeyboardInterrupt()
            command=["gz","service","-s",f"/world/{manifest['world']}/set_pose","--reqtype","gz.msgs.Pose",
                     "--reptype","gz.msgs.Boolean","--timeout","3000","--req",
                     'name:"robot", position:{x:1.1,y:6.2,z:0.101}, orientation:{w:1}']
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=5)
            if result.returncode==0 and "true" in result.stdout:
                ready=True;break
        if not ready:raise RuntimeError("Gazebo never accepted the declared route start pose")
        config_path=None
        if a.camera_blackout:
            config=json.loads(Path("configs/robot_sim/localization.json").read_text())
            config["topics"]["rgb"]="/fault_camera/left/image"
            config_path=root/"fault_localization_config.json"
            config_path.write_text(json.dumps(config,indent=2)+"\n")
        ace=["/data/fyp_3d/envs/ace/bin/python","-m","research3d.robot_sim.ace_worker","--binding",str(a.ace_binding.resolve()),
             "--online-root",str(run/"queries"),"--source","/data/fyp_3d/sources/ace"]
        local=[sys.executable,"-m","research3d.robot_sim.run_localization","--map-manifest",str(a.world_manifest.resolve()),
               "--output",str(run),"--duration-wall-s",str(a.seconds*3+45),"--domain-id",str(a.domain_id),
               "--ace-command-json",json.dumps(ace)]
        if a.spvloc_command_json:local.extend(["--spvloc-command-json",a.spvloc_command_json])
        if a.allow_spvloc_corrections:local.append("--allow-spvloc-corrections")
        if config_path:local.extend(["--config",str(config_path)])
        if a.spvloc_references:
            libraries="/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux:"+env.get("LD_LIBRARY_PATH","")
            spv=["/usr/bin/env","LD_LIBRARY_PATH="+libraries,"PYOPENGL_PLATFORM=egl","MPLCONFIGDIR="+str(root/"matplotlib"),
                 "/data/fyp_3d/envs/semantic_loc/bin/python","-m","research3d.robot_sim.spvloc_worker",
                 "--references",str(a.spvloc_references.resolve()),"--online-root",str(run/"queries"),
                 "--dataset-root",str(a.mapping_dataset.resolve()),"--checkpoint",str(a.spvloc_checkpoint.resolve())]
            local.extend(["--spvloc-command-json",json.dumps(spv)])
        localization=launch("localization_launcher",local)
        ready_deadline=time.monotonic()+120.
        while not (run/"workers_ready.json").exists():
            if interrupted:raise KeyboardInterrupt()
            if localization.poll() is not None:raise RuntimeError("localization exited before actual model readiness")
            if any(c.poll() is not None for c in children):raise RuntimeError("simulation component exited during model startup")
            if time.monotonic()>ready_deadline:raise TimeoutError("actual model readiness handshake exceeded 120 seconds")
            time.sleep(.2)
        if a.camera_blackout:
            launch("camera_fault",[sys.executable,"-m","research3d.robot_sim.camera_fault","--start-s",str(a.camera_blackout[0]),
                 "--end-s",str(a.camera_blackout[1]),"--output",str(root/"camera_fault_events.jsonl")])
        capture_command=[sys.executable,"-m","research3d.robot_sim.collect","query","--world-manifest",str(a.world_manifest.resolve()),
              "--output",str(root/"capture"),"--seconds",str(a.seconds),"--image-hz","10","--commands",str(a.route.resolve())]
        if a.events:capture_command.extend(["--events",str(a.events.resolve())])
        if a.camera_blackout:capture_command.extend(["--image-topic","/fault_camera/left/image","--right-image-topic","/fault_camera/right/image"])
        route_began=time.monotonic()
        recording=launch("collector",capture_command)
        deadline=time.monotonic()+a.seconds*5+90
        while not interrupted and recording.poll() is None and time.monotonic()<deadline:
            if any(c.poll() is not None for c in children[:-1]):raise RuntimeError("a required simulation component exited")
            time.sleep(.2)
        if interrupted:raise KeyboardInterrupt()
        if recording.poll()!=0:raise RuntimeError("trajectory did not complete; see collector.log")
        route_completed=time.monotonic()
    finally:
        for name,child in reversed(list(zip(child_names,children))):
            if child.poll() is None:
                # The ROS launcher/bridge implement their own ordered cleanup;
                # do not interrupt their children a second time during finally.
                if name=="gazebo":os.killpg(child.pid,signal.SIGINT)
                else:child.send_signal(signal.SIGINT)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
        for handle in logs:handle.close()
        (root/"launcher_receipt.json").write_text(json.dumps(dict(child_pids=[c.pid for c in children],
            child_names=child_names,exit_codes=[c.returncode for c in children],domain_id=a.domain_id,GZ_PARTITION=env["GZ_PARTITION"],
            elapsed_wall_s=time.monotonic()-began,
            capture_wall_s=None if route_completed is None else route_completed-route_began,
            requested_simulation_duration_s=a.seconds,
            interrupted=interrupted,cleanup="only process groups created by this launcher"),indent=2)+"\n")
    fault_receipt=root/"camera_fault_events.summary.json"
    if fault_receipt.exists():shutil.copy2(fault_receipt,root/"capture/offline_gt/camera_fault.json")
    subprocess.run([sys.executable,"-m","research3d.robot_sim.evaluate","--dataset",str(root/"capture"),
                    "--run",str(run),"--output",str(root/"evaluation")],check=True)


if __name__=="__main__":main()
