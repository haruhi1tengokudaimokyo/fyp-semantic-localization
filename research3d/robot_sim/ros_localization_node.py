"""ROS 2 sensor bridge and asynchronous ACE global localization.

Run with ROS Jazzy sourced. The official robot_localization ekf_node runs as a
separate process using configs/robot_sim/ekf.yaml. This node never subscribes to
Gazebo model/world pose or evaluation GT. Camera-info must arrive before ACE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .localization import GlobalLocalizer, LocalizationConfig, differential_wheel_velocity
from .scheduler import AceRpcProcess, LatestOnlyWorker, QueryTrigger, ExecutionAlternator


def stamp_sec(stamp):
    return stamp.sec+stamp.nanosec/1e9


def transform_from_ros_pose(value):
    q = value.orientation
    out = np.eye(4)
    out[:3,:3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    out[:3,3] = [value.position.x,value.position.y,value.position.z]
    return out


def rgb_from_message(message):
    channels = {"rgb8":3,"bgr8":3,"rgba8":4,"bgra8":4,"mono8":1}
    if message.encoding not in channels:
        raise ValueError(f"unsupported image encoding {message.encoding}")
    c = channels[message.encoding]
    if message.step < message.width*c or len(message.data) != message.height*message.step:
        raise ValueError("invalid image step/data size")
    raw = np.frombuffer(bytes(message.data),dtype=np.uint8).reshape(message.height,message.step)
    pixels = raw[:,:message.width*c].reshape(message.height,message.width,c)
    if c==1:
        return np.repeat(pixels,3,axis=2)
    if message.encoding in {"bgr8","bgra8"}:
        return pixels[:,:,:3][:,:,::-1].copy()
    return pixels[:,:,:3].copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/robot_sim/localization.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ace-command-json",help="JSON argv list; omit for explicit local-only run")
    parser.add_argument("--spvloc-command-json",help="Optional official SPVLoc recovery RPC argv")
    parser.add_argument("--allow-spvloc-corrections",action="store_true",
                        help="Experimental ablation only: permit unqualified SPVLoc candidates through the original recovery gate")
    parser.add_argument("--duration-wall-s",type=float,default=0.)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    # A JSON config alone must never silently promote this unqualified model.
    cfg["global"]["allow_spvloc_corrections"]=args.allow_spvloc_corrections
    if (args.ace_command_json or args.spvloc_command_json) and len(cfg["global"].get("map_sha256") or "") != 64:
        raise ValueError("ACE online mode requires exact map SHA256 binding; use run_localization launcher")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"queries").mkdir(exist_ok=True)
    # Imports remain lazy: the contracts and tests run without installing ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState, Imu, Image, CameraInfo
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

    def fill_transform(message,matrix):
        message.transform.translation.x=float(matrix[0,3])
        message.transform.translation.y=float(matrix[1,3])
        message.transform.translation.z=float(matrix[2,3])
        q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
        (message.transform.rotation.x,message.transform.rotation.y,
         message.transform.rotation.z,message.transform.rotation.w)=map(float,q)

    class RobotLocalization(Node):
        def __init__(self):
            super().__init__("robot_global_localization")
            self.core = GlobalLocalizer(LocalizationConfig(**cfg["global"]),cfg["T_base_camera_optical"])
            self.trigger = QueryTrigger(**cfg["query_trigger"])
            self.responses = queue.Queue()
            self.rpc = self.spv_rpc = self.worker = None
            self.selector = ExecutionAlternator(bool(args.ace_command_json), bool(args.spvloc_command_json))
            self.camera = None
            self.last_joint = None
            self.last_imu_s = None
            self.counts={"joint_messages":0,"wheel_messages":0,"imu_messages":0,"local_poses":0,
                         "camera_messages":0,"queries_submitted":0,"candidate_messages":0,
                         "invalid_input":0}
            self.frame_seq=0
            self.tf_broadcaster=TransformBroadcaster(self)
            self.static_broadcaster=StaticTransformBroadcaster(self)
            for child,matrix in [("camera_optical_frame",cfg["T_base_camera_optical"]),("imu_link",np.eye(4))]:
                transform=TransformStamped()
                transform.header.stamp=self.get_clock().now().to_msg()
                transform.header.frame_id="base_link"
                transform.child_frame_id=child
                fill_transform(transform,np.asarray(matrix))
                self.static_broadcaster.sendTransform(transform)
            self.wheel_pub=self.create_publisher(Odometry,"/wheel/odometry",10)
            self.imu_pub=self.create_publisher(Imu,"/imu/data",10)
            self.candidate_pub=self.create_publisher(String,"/ace/candidate",10)
            self.spv_candidate_pub=self.create_publisher(String,"/spvloc/candidate",10)
            self.status_pub=self.create_publisher(String,"/localization/status",10)
            self.create_subscription(JointState,cfg["topics"]["joint_states"],self.joints,qos_profile_sensor_data)
            self.create_subscription(Imu,cfg["topics"]["imu"],self.imu,qos_profile_sensor_data)
            self.create_subscription(Image,cfg["topics"]["rgb"],self.image,qos_profile_sensor_data)
            self.create_subscription(CameraInfo,cfg["topics"]["camera_info"],self.camera_info,qos_profile_sensor_data)
            self.create_subscription(Odometry,"/odometry/filtered",self.odometry,10)
            self.create_subscription(String,"/ace/candidate",self.candidate,10)
            self.create_subscription(String,"/spvloc/candidate",self.candidate,10)
            self.timer=self.create_timer(1./30.,self.tick)
            self.trace=(args.output/"localization.jsonl").open("w",buffering=1)
            self.decisions=(args.output/"ace_decisions.jsonl").open("w",buffering=1)
            self.queries=(args.output/"queries.jsonl").open("w",buffering=1)
            ready_receipts={}
            try:
                for branch,command in [("ace",args.ace_command_json),("spvloc",args.spvloc_command_json)]:
                    if not command:continue
                    rpc=AceRpcProcess(json.loads(command),args.output/(branch+"_worker.log"))
                    if branch=="ace":self.rpc=rpc
                    else:self.spv_rpc=rpc
                    ready=rpc.wait_ready()
                    if (ready.get("map_id"),ready.get("map_sha256"))!=(cfg["global"]["map_id"],cfg["global"]["map_sha256"]):
                        raise ValueError("worker ready map binding mismatch")
                    ready_receipts[branch]=ready
            except BaseException:
                for rpc in (self.rpc,self.spv_rpc):
                    if rpc is not None:rpc.close()
                raise
            if self.rpc is not None or self.spv_rpc is not None:
                def dispatch(item):
                    # Pending images may be replaced many times while a model
                    # runs. Alternating on submissions could starve ACE forever.
                    # Decide only when the one worker actually starts an RPC.
                    branch=self.selector.choose(self.core.T_map_odom is None)
                    item["_localizer"]=branch
                    request=dict(item)
                    request.pop("_localizer")
                    submitted=request.pop("_submitted_monotonic_s")
                    started=time.monotonic()
                    try:
                        result=(self.spv_rpc if branch=="spvloc" else self.rpc)(request)
                    except Exception as exc:
                        result={"status":"error","error":f"{type(exc).__name__}: {exc}",
                                "request_id":request["request_id"],"query_id":request["query_id"],
                                "map_id":request["map_id"],"map_sha256":request["map_sha256"],
                                "stamp_sec":request["timestamp_ns"]/1e9,
                                "source":"official_spvloc" if branch=="spvloc" else "official_ace_dsacstar"}
                    returned=time.monotonic()
                    result["rpc_wall_seconds"]=returned-started
                    result["runtime_timing"]={"dispatched_localizer":branch,"capture_ros_time_s":request["timestamp_ns"]/1e9,
                        "rpc_return_ros_time_s":self.get_clock().now().nanoseconds/1e9,
                        "pending_wait_wall_s":started-submitted,"rpc_roundtrip_wall_s":returned-started,
                        "submit_to_return_wall_s":returned-submitted}
                    return result
                # Exactly one executing RPC and one replacement pending slot
                # shared across ACE/SPVLoc: no simultaneous GPU inference.
                self.worker=LatestOnlyWorker(dispatch,self.responses.put)
            protocol={"config":cfg,"local_algorithm":"official_robot_localization_EKF",
                      "global_algorithms":(["official_ACE_DSACStar"] if self.rpc else [])+(["official_SPVLoc"] if self.spv_rpc else []),
                      "ace_enabled":self.rpc is not None,"spvloc_enabled":self.spv_rpc is not None,"macvo_enabled":False,
                      "spvloc_corrections_allowed":args.allow_spvloc_corrections,
                      "spvloc_policy":"experimental_original_recovery_gate" if args.allow_spvloc_corrections else "shadow_unqualified_model",
                      "spvloc_qualification_note":"Current checkpoint failed fixed-map development localization. Default real inference records candidates but cannot alter map->odom; opt-in is an explicit ablation, not qualification.",
                      "global_schedule":"at_most_1Hz_total; alternate actual ACE/SPV executions when globally unknown; ACE tracking updates; one latest pending slot",
                      "timestamp_policy":"capture time from image header; RPC return ROS and monotonic wall times measured independently; callback now used for conservative stale/history gates",
                      "wheel_source":"simulated_joint_encoder_velocity_not_world_odometry",
                      "imu_fused_fields":["angular_velocity.z"],"wheel_fused_fields":["linear.x","angular.z"],
                      "robot_motion_constraint":{"linear.y":0.,"standard_deviation_m_s":cfg["wheel"]["lateral_constraint_std_m_s"],
                                                 "type":"finite_uncertainty_nonholonomic_differential_drive"},
                      "online_gt_used":False,"ace_pose_covariance_available":False,
                      "update_policy":"gated_map_to_odom_correction_not_pose_graph",
                      "parameters_chosen_without_test_ground_truth":True,
                      "measurement_covariance_note":"declared engineering sensor uncertainty, not calibrated from truth"}
            (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
            ready_path=args.output/"workers_ready.json"
            partial=ready_path.with_suffix(".tmp")
            partial.write_text(json.dumps({"status":"WORKERS_READY","receipts":ready_receipts,
                                          "local_only":not ready_receipts,"map_id":cfg["global"]["map_id"]},indent=2)+"\n")
            partial.replace(ready_path)

        def joints(self,msg):
            self.counts["joint_messages"]+=1
            wheel=cfg["wheel"]
            try:
                left=msg.name.index(wheel["left_joint"])
                right=msg.name.index(wheel["right_joint"])
                stamp=stamp_sec(msg.header.stamp)
                if self.last_joint is not None and stamp<=self.last_joint[0]:
                    return
                # Use only encoder joint rates; never Gazebo world/model pose.
                if len(msg.velocity)>max(left,right):
                    vl,vr=float(msg.velocity[left]),float(msg.velocity[right])
                elif len(msg.position)>max(left,right):
                    current=np.array([msg.position[left],msg.position[right]])
                    if self.last_joint is None:
                        self.last_joint=(stamp,current)
                        return
                    dt=stamp-self.last_joint[0]
                    if dt<=0 or dt>.2:
                        self.last_joint=(stamp,current)
                        return
                    # Continuous wheel joints are expected to be unwrapped.
                    vl,vr=(current-self.last_joint[1])/dt
                else:
                    raise ValueError("wheel JointState has neither rates nor positions")
                positions=np.array([msg.position[left],msg.position[right]]) if len(msg.position)>max(left,right) else np.zeros(2)
                self.last_joint=(stamp,positions)
                forward,yaw=differential_wheel_velocity(vl,vr,wheel["radius_m"],wheel["separation_m"])
                odom=Odometry()
                odom.header.stamp=msg.header.stamp
                odom.header.frame_id="odom"
                odom.child_frame_id="base_link"
                odom.pose.pose.orientation.w=1.
                odom.pose.covariance=[1e6 if i%7==0 else 0. for i in range(36)]
                odom.twist.twist.linear.x=forward
                odom.twist.twist.linear.y=0.
                odom.twist.twist.angular.z=yaw
                odom.twist.covariance=[1e6 if i%7==0 else 0. for i in range(36)]
                odom.twist.covariance[0]=wheel["forward_std_m_s"]**2
                odom.twist.covariance[7]=wheel["lateral_constraint_std_m_s"]**2
                odom.twist.covariance[35]=wheel["yaw_std_rad_s"]**2
                self.wheel_pub.publish(odom)
                self.counts["wheel_messages"]+=1
            except (ValueError,IndexError,TypeError):
                self.counts["invalid_input"]+=1

        def imu(self,msg):
            if msg.header.frame_id!=cfg["expected_frames"]["imu"]:
                self.counts["invalid_input"]+=1
                return
            stamp=stamp_sec(msg.header.stamp)
            if self.last_imu_s is not None and stamp<=self.last_imu_s:
                return
            if not np.isfinite([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]).all():
                self.counts["invalid_input"]+=1
                return
            self.last_imu_s=stamp
            output=Imu()
            output.header.stamp=msg.header.stamp
            output.header.frame_id="imu_link"
            output.orientation_covariance[0]=-1.
            output.linear_acceleration_covariance[0]=-1.
            output.angular_velocity=msg.angular_velocity
            output.angular_velocity_covariance=[0.]*9
            for i in [0,4,8]:
                output.angular_velocity_covariance[i]=cfg["imu_gyro_std_rad_s"]**2
            self.imu_pub.publish(output)
            self.counts["imu_messages"]+=1

        def odometry(self,msg):
            try:
                stamp=stamp_sec(msg.header.stamp)
                if msg.header.frame_id!="odom" or msg.child_frame_id!="base_link":
                    raise ValueError("unexpected local odometry frames")
                covariance=msg.pose.covariance
                self.core.add_odometry(stamp,transform_from_ros_pose(msg.pose.pose),
                                       covariance[0]+covariance[7]+covariance[35])
                self.counts["local_poses"]+=1
            except ValueError:
                self.counts["invalid_input"]+=1

        def camera_info(self,msg):
            if msg.header.frame_id!=cfg["expected_frames"]["camera"]:
                self.counts["invalid_input"]+=1
                return
            k=np.asarray(msg.k,dtype=float).reshape(3,3)
            if not np.isfinite(k).all() or msg.width<=0 or msg.height<=0 or k[0,0]<=0 or k[1,1]<=0:
                self.counts["invalid_input"]+=1
                return
            self.camera={"width":int(msg.width),"height":int(msg.height),"K":k.tolist()}

        def image(self,msg):
            self.counts["camera_messages"]+=1
            if msg.header.frame_id!=cfg["expected_frames"]["camera"]:
                self.counts["invalid_input"]+=1
                return
            if self.worker is None or self.camera is None:
                return
            now=self.get_clock().now().nanoseconds/1e9
            stamp=stamp_sec(msg.header.stamp)
            # A delayed transport frame cannot create a fresh inference request.
            if now-stamp>cfg["max_query_transport_age_s"] or stamp>now+.01:
                self.counts["invalid_input"]+=1
                return
            try:
                if msg.width!=self.camera["width"] or msg.height!=self.camera["height"]:
                    raise ValueError("camera info/image dimensions mismatch")
                rgb=rgb_from_message(msg)
                execute,reason=self.trigger.decide(rgb,stamp,self.core.snapshot(now))
            except ValueError:
                self.counts["invalid_input"]+=1
                return
            if not execute:
                return
            self.frame_seq+=1
            qid=f"rgb_{msg.header.stamp.sec}_{msg.header.stamp.nanosec:09d}_{self.frame_seq:06d}"
            path=(args.output/"queries"/(qid+".png")).resolve()
            temporary=path.with_suffix(".partial.png")
            if not cv2.imwrite(str(temporary),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                self.counts["invalid_input"]+=1
                return
            temporary.replace(path)
            request={"op":"localize","request_id":qid,"query_id":qid,
                     "map_id":cfg["global"]["map_id"],
                     "map_sha256":cfg["global"]["map_sha256"],
                     "timestamp_ns":msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,
                     "image_path":str(path),"camera":self.camera.copy(),
                     "metadata":{"sequence_id":cfg["sequence_id"],"sensor_frame":"camera_optical_frame"}}
            request["_localizer"]="select_on_execution"
            request["_submitted_monotonic_s"]=time.monotonic()
            self.queries.write(json.dumps(request)+"\n")
            self.worker.submit(request)
            self.counts["queries_submitted"]+=1

        def candidate(self,msg):
            self.counts["candidate_messages"]+=1
            now=self.get_clock().now().nanoseconds/1e9
            try:
                row=json.loads(msg.data)
                before=None if self.core.T_map_odom is None else self.core.T_map_odom.tolist()
                decision=(self.core.observe_candidates(row,now) if row.get("source")=="official_spvloc" else self.core.observe(row,now))
                after=None if self.core.T_map_odom is None else self.core.T_map_odom.tolist()
                self.decisions.write(json.dumps({"received_time_s":now,"candidate":row,"decision":decision,
                    "T_map_odom_before":before,"T_map_odom_after":after},allow_nan=False)+"\n")
            except (ValueError,TypeError):
                self.counts["invalid_input"]+=1

        def tick(self):
            while True:
                try:
                    row=self.responses.get_nowait()
                except queue.Empty:
                    break
                msg=String()
                msg.data=json.dumps(row,allow_nan=False)
                (self.spv_candidate_pub if row.get("source")=="official_spvloc" else self.candidate_pub).publish(msg)
            now=self.get_clock().now().nanoseconds/1e9
            status=self.core.snapshot(now)
            status.update(counts=self.counts.copy(),ace_enabled=self.rpc is not None,
                          spvloc_available=self.spv_rpc is not None,
                          backend="wheel_imu_ekf_plus_gated_absolute_map_odom",
                          actual_localizer_executions=self.selector.started.copy(),
                          query_trigger_counts=self.trigger.counts.copy(),
                          pending_queries_replaced=0 if self.worker is None else self.worker.overwritten)
            if status["global_valid"]:
                transform=TransformStamped()
                transform.header.stamp=self.get_clock().now().to_msg()
                transform.header.frame_id="map"
                transform.child_frame_id="odom"
                fill_transform(transform,np.asarray(status["T_map_odom"]))
                self.tf_broadcaster.sendTransform(transform)
            msg=String()
            msg.data=json.dumps(status,allow_nan=False)
            self.status_pub.publish(msg)
            self.trace.write(msg.data+"\n")

        def finish(self):
            if self.rpc is not None:
                self.rpc.close()
            if self.spv_rpc is not None:
                self.spv_rpc.close()
            if self.worker is not None:
                self.worker.close()
            last=self.core.snapshot(self.get_clock().now().nanoseconds/1e9)
            last.update(spvloc_available=self.spv_rpc is not None,backend="wheel_imu_ekf_plus_gated_absolute_map_odom")
            summary={"counts":self.counts,"last_status":last,
                     "ace_enabled":self.rpc is not None,"spvloc_enabled":self.spv_rpc is not None,"query_trigger_counts":self.trigger.counts,
                     "pending_queries_replaced":0 if self.worker is None else self.worker.overwritten}
            (args.output/"runtime_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            self.trace.close()
            self.decisions.close()
            self.queries.close()

    rclpy.init(args=["--ros-args","-p","use_sim_time:=true"])
    node=RobotLocalization()
    began=time.monotonic()
    try:
        while rclpy.ok() and (args.duration_wall_s<=0 or time.monotonic()-began<args.duration_wall_s):
            rclpy.spin_once(node,timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()
