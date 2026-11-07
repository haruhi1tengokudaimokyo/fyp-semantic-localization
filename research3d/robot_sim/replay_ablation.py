"""Bounded 1x raw-sensor ablation using the existing ROS EKF/RPC/replay tools.

Default is EKF-only and starts no model/GPU process. This does not create a
Gazebo world or actuator commands. A capture must contain actual wheel/IMU
logs; sensor counters or query poses are never substituted for missing inputs.
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

from research3d.fusion.contracts import write_json
from research3d.robot_sim.replay_sensors import load_recording


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=["ekf","ace","ace_spv_shadow","ace_spv_experimental"],default="ekf")
    p.add_argument("--ace-binding",type=Path)
    p.add_argument("--spvloc-references",type=Path)
    p.add_argument("--spvloc-checkpoint",type=Path,default=Path("/data/fyp_3d/runs/semantic_localization_20261103/spvloc_100_20_20_float_targets_v2/checkpoints/epochepoch=18-stepstep=6517.ckpt"))
    p.add_argument("--domain-id",type=int,default=74)
    p.add_argument("--ready-timeout-s",type=float,default=120.)
    a=p.parse_args(argv)
    if a.domain_id in {71,72} or not 1<=a.domain_id<=232:
        p.error("use an isolated domain distinct from live world 71 and demo 72")
    if not 1<=a.ready_timeout_s<=180:p.error("ready timeout must be bounded to 1-180 seconds")
    if a.mode!="ekf" and not a.ace_binding:p.error("ACE modes require --ace-binding")
    if a.mode.startswith("ace_spv") and not a.spvloc_references:p.error("SPVLoc modes require --spvloc-references")
    if a.output.exists():raise FileExistsError("fresh output required; existing runs are immutable")
    dataset,manifest,_,receipt=load_recording(a.dataset)
    if a.mode!="ekf":
        binding=json.loads(a.ace_binding.read_text())
        if (binding["map_id"],binding["map_sha256"])!=(manifest["map_id"],manifest["map_sha256"]):
            raise ValueError("ACE model belongs to a different fixed map")
    if a.mode.startswith("ace_spv"):
        binding=json.loads((a.spvloc_references/"reference_binding.json").read_text())
        if (binding["map_id"],binding["map_sha256"])!=(manifest["map_id"],manifest["map_sha256"]):
            raise ValueError("SPVLoc references belong to a different fixed map")
    import rclpy  # fail before starting children when ROS Jazzy is not sourced
    del rclpy
    output=a.output.resolve();output.mkdir(parents=True)
    local=output/"localization"
    env=os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(a.domain_id),ROS_LOG_DIR=str(output/"ros_logs"))
    duration=receipt["recorded_duration_s"]
    command=[sys.executable,"-m","research3d.robot_sim.run_localization","--map-manifest",str(dataset/"map_manifest.json"),
             "--output",str(local),"--domain-id",str(a.domain_id),"--duration-wall-s",str(a.ready_timeout_s+duration+30.)]
    if a.mode!="ekf":
        ace=["/data/fyp_3d/envs/ace/bin/python","-m","research3d.robot_sim.ace_worker","--binding",str(a.ace_binding.resolve()),
             "--online-root",str(local/"queries"),"--source","/data/fyp_3d/sources/ace"]
        command.extend(["--ace-command-json",json.dumps(ace)])
    if a.mode.startswith("ace_spv"):
        libraries="/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux:"+env.get("LD_LIBRARY_PATH","")
        spv=["/usr/bin/env","LD_LIBRARY_PATH="+libraries,"PYOPENGL_PLATFORM=egl","MPLCONFIGDIR="+str(output/"matplotlib"),
             "/data/fyp_3d/envs/semantic_loc/bin/python","-m","research3d.robot_sim.spvloc_worker",
             "--references",str(a.spvloc_references.resolve()),"--online-root",str(local/"queries"),
             "--dataset-root",str(dataset),"--checkpoint",str(a.spvloc_checkpoint.resolve())]
        command.extend(["--spvloc-command-json",json.dumps(spv)])
    if a.mode=="ace_spv_experimental":command.append("--allow-spvloc-corrections")
    replay=[sys.executable,"-m","research3d.robot_sim.replay_sensors","--dataset",str(dataset),
            "--receipt",str(output/"replay_receipt.json"),"--domain-id",str(a.domain_id),"--rate","1","--start-delay-s","3"]
    protocol=dict(mode=a.mode,recording=receipt,replay_rate=1.,start_models_before_replay=True,
                  model_rpc_enabled=a.mode!="ekf",spvloc_corrections_allowed=a.mode=="ace_spv_experimental",
                  domain_id=a.domain_id,commands=[command,replay],published_GT=False,
                  scope="same recorded sensors with measured online model latency; no live physics/control and no new benchmark capture")
    write_json(output/"protocol.json",protocol)
    interrupted=False;children=[];logs=[];error=None
    def interrupt(signum,frame):
        nonlocal interrupted
        interrupted=True
    original={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    def launch(name,argv):
        log=(output/(name+".log")).open("w");logs.append(log)
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        children.append((name,child));return child
    began=time.monotonic()
    try:
        estimator=launch("estimator_launcher",command)
        ready_file=local/"workers_ready.json"
        deadline=time.monotonic()+a.ready_timeout_s
        while not ready_file.exists():
            if interrupted:raise KeyboardInterrupt()
            if estimator.poll() is not None:raise RuntimeError("estimator exited before real workers became ready")
            if time.monotonic()>=deadline:raise TimeoutError("workers-ready deadline exceeded")
            time.sleep(.05)
        ready=json.loads(ready_file.read_text())
        if ready["status"]!="WORKERS_READY" or ready["map_id"]!=manifest["map_id"]:
            raise ValueError("invalid ready receipt")
        if set(ready["receipts"]) != (set() if a.mode=="ekf" else {"ace","spvloc"} if a.mode.startswith("ace_spv") else {"ace"}):
            raise ValueError("ready models differ from requested ablation")
        sensor_replay=launch("sensor_replay",replay)
        deadline=time.monotonic()+duration+25.
        while sensor_replay.poll() is None:
            if interrupted:raise KeyboardInterrupt()
            if estimator.poll() is not None:raise RuntimeError("estimator stopped during recorded-sensor replay")
            if time.monotonic()>=deadline:raise TimeoutError("1x recorded-sensor replay exceeded bounded deadline")
            time.sleep(.05)
        if sensor_replay.returncode!=0:raise RuntimeError("recorded-sensor replay failed; see preserved logs")
        time.sleep(.15)  # drain already-published ROS callbacks, without inventing future sensor samples
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for name,child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=15.)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=4.)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3.)
        for log in logs:log.close()
        for sig,handler in original.items():signal.signal(sig,handler)
        write_json(output/"launcher_receipt.json",dict(child_pids=[c.pid for _,c in children],
                   exit_codes={name:c.returncode for name,c in children},error=error,interrupted=interrupted,
                   elapsed_wall_s=time.monotonic()-began,cleanup="only child process groups launched by this invocation"))
    # Independent offline GT is read only after estimator/replay have stopped.
    from research3d.robot_sim.evaluate import evaluate
    metrics=evaluate(dataset,local,output/"evaluation")
    print(json.dumps(dict(status="RECORDED_SENSOR_ABLATION_EVALUATED",mode=a.mode,output=str(output),
                         evaluated=metrics["samples"],absolute=metrics["absolute"]),indent=2))


if __name__=="__main__":main()
