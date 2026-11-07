"""GT-free ROS 2 replay of captured encoders, gyro and synchronized stereo.

This is recorded-sensor playback, not a second live Gazebo run. No GT, pose
label, odometry, transform, or velocity command topic is published. The default
isolated domain 72 prevents mixing recorded inputs with the live domain 71.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image as PILImage

from research3d.fusion.contracts import no_truth,read_jsonl,sha256_file,write_json


def stamp_parts(ns):
    if type(ns) is not int or ns<0: raise ValueError('nonnegative integer timestamp_ns required')
    return divmod(ns,1_000_000_000)


def _vector(values,size):
    if len(values)!=size or not np.isfinite(np.asarray(values,float)).all():
        raise ValueError('invalid finite sensor vector')
    return [float(x) for x in values]


def validate_sensor(row,kind):
    no_truth(row);stamp_parts(row['timestamp_ns'])
    if kind=='imu':
        if set(row)-{'timestamp_ns','frame_id','angular_velocity','linear_acceleration','angular_velocity_covariance','orientation_excluded'}:
            raise ValueError('IMU replay accepts recorded gyro/acceleration only')
        if row.get('orientation_excluded') is not True: raise ValueError('absolute simulator orientation must be excluded')
        _vector(row['angular_velocity'],3);_vector(row['linear_acceleration'],3)
        _vector(row['angular_velocity_covariance'],9)
        if row['frame_id']!='base_link': raise ValueError('unknown IMU frame')
    elif kind=='joint_states':
        if set(row)-{'timestamp_ns','names','positions','velocities'}: raise ValueError('unknown encoder fields')
        if len(row['names'])!=2 or set(row['names'])!={'left_wheel_joint','right_wheel_joint'}:
            raise ValueError('both real wheel joints required')
        _vector(row['positions'],2);_vector(row['velocities'],2)
    else: raise ValueError('unknown sensor kind')
    return row


def merge_events(images,joints,imu):
    streams={'stereo':images,'joint_states':joints,'imu':imu}
    priority={'joint_states':0,'imu':1,'stereo':2}
    events=[]
    for kind,rows in streams.items():
        previous=-1
        for row in rows:
            t=row['timestamp_ns'];stamp_parts(t)
            if t<=previous: raise ValueError('timestamps must strictly increase within '+kind)
            previous=t
            if kind=='stereo':
                no_truth(row)
                allowed={'query_id','timestamp_ns','image_path','image_sha256','camera','map_id',
                         'right_image_path','right_image_sha256','right_timestamp_ns'}
                if set(row)-allowed:raise ValueError('unexpected query metadata: no pose labels accepted')
                if row['right_timestamp_ns']!=t: raise ValueError('asynchronous stereo pair')
            else: validate_sensor(row,kind)
            events.append((t,kind,row))
    return sorted(events,key=lambda x:(x[0],priority[x[1]]))


def load_recording(root):
    root=Path(root).resolve(strict=True)
    paths=[root/'map_manifest.json',root/'query/frames.jsonl',root/'query/sensors/joint_states.jsonl',root/'query/sensors/imu.jsonl']
    manifest=json.loads(paths[0].read_text())
    images,joints,imu=[read_jsonl(path) for path in paths[1:]]
    if len(images)<3 or len(joints)<3 or len(imu)<3: raise ValueError('incomplete recorded sensor streams')
    baseline=float(manifest['calibration']['stereo_baseline_m'])
    if baseline<=0: raise ValueError('positive measured stereo baseline required')
    for row in images:
        if row['map_id']!=manifest['map_id']: raise ValueError('mixed map input')
        if row['camera']!=manifest['camera']: raise ValueError('camera calibration changed')
        for key,checksum in [('image_path','image_sha256'),('right_image_path','right_image_sha256')]:
            path=(root/row[key]).resolve(strict=True)
            if not path.is_relative_to(root/'query'): raise ValueError('image outside query split')
            if sha256_file(path)!=row[checksum]: raise ValueError('changed RGB file')
            with PILImage.open(path) as image:
                image.verify()
                if image.size!=(manifest['camera']['width'],manifest['camera']['height']):
                    raise ValueError('image size/calibration mismatch')
    events=merge_events(images,joints,imu)
    first,last=images[0]['timestamp_ns'],images[-1]['timestamp_ns']
    for name,rows in [('imu',imu),('joint_states',joints)]:
        if rows[0]['timestamp_ns']>first+20_000_000 or rows[-1]['timestamp_ns']<last-20_000_000:
            raise ValueError(name+' does not cover the image trajectory')
    # Retain pre-image sensor samples for EKF warm-up; never manufacture samples.
    receipt=dict(input_sha256={str(p):sha256_file(p) for p in paths},map_id=manifest['map_id'],
                 counts=dict(stereo=len(images),joint_states=len(joints),imu=len(imu)),
                 first_timestamp_ns=events[0][0],last_timestamp_ns=events[-1][0],
                 recorded_duration_s=(events[-1][0]-events[0][0])/1e9,
                 max_gap_s={name:max((b['timestamp_ns']-a['timestamp_ns'])/1e9 for a,b in zip(rows,rows[1:]))
                            for name,rows in [('imu',imu),('joint_states',joints),('stereo',images)]},
                 GT_read=False,source='recorded Gazebo sensors; not live physics or physical-robot data')
    return root,manifest,events,receipt


def sensor_message(kind,row,message_class):
    """Construct actual ROS message classes; injectable classes permit CPU tests."""
    validate_sensor(row,kind);msg=message_class()
    msg.header.stamp.sec,msg.header.stamp.nanosec=stamp_parts(row['timestamp_ns'])
    if kind=='imu':
        msg.header.frame_id=row['frame_id']
        msg.orientation.w=1.;msg.orientation_covariance=[-1.,0.,0.,0.,0.,0.,0.,0.,0.]
        for obj,key in [(msg.angular_velocity,'angular_velocity'),(msg.linear_acceleration,'linear_acceleration')]:
            obj.x,obj.y,obj.z=_vector(row[key],3)
        msg.angular_velocity_covariance=_vector(row['angular_velocity_covariance'],9)
    else:
        msg.header.frame_id='base_link';msg.name=list(row['names'])
        msg.position=_vector(row['positions'],2);msg.velocity=_vector(row['velocities'],2)
    return msg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--domain-id',type=int,default=72)
    p.add_argument('--rate',type=float,default=1.)
    p.add_argument('--start-delay-s',type=float,default=3.)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if args.domain_id==71 or not 0<=args.domain_id<=232: p.error('use an isolated domain, never live Gazebo domain71')
    if not math.isfinite(args.rate) or args.rate<=0 or args.rate>4: p.error('rate must be positive and <=4')
    if not math.isfinite(args.start_delay_s) or args.start_delay_s<0 or args.start_delay_s>30: p.error('invalid startup delay')
    if args.receipt.exists(): raise FileExistsError('refusing to overwrite replay receipt')
    root,manifest,events,receipt=load_recording(args.dataset)
    if args.validate_only:
        write_json(args.receipt,dict(receipt,status='INPUT_VALIDATED_NOT_REPLAYED'));return
    os.environ['ROS_DOMAIN_ID']=str(args.domain_id)
    import rclpy
    from sensor_msgs.msg import Image,CameraInfo,Imu,JointState
    from rosgraph_msgs.msg import Clock
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init();node=rclpy.create_node('fyp_recorded_sensor_replay_no_gt')
    clock=node.create_publisher(Clock,'/clock',10)
    publishers={kind:node.create_publisher(cls,topic,qos_profile_sensor_data) for kind,cls,topic in [
        ('imu',Imu,'/robot/imu'),('joint_states',JointState,'/robot/joint_states'),
        ('left',Image,'/robot/camera/image'),('right',Image,'/robot/camera_right/image'),
        ('left_info',CameraInfo,'/robot/camera/camera_info'),('right_info',CameraInfo,'/robot/camera_right/camera_info')]}
    def publish_clock(t):
        msg=Clock();msg.clock.sec,msg.clock.nanosec=stamp_parts(t);clock.publish(msg)
    def camera_info(t,side):
        msg=CameraInfo();msg.header.stamp.sec,msg.header.stamp.nanosec=stamp_parts(t)
        msg.header.frame_id='camera_optical_frame' if side=='left' else 'camera_right_optical_frame'
        c=manifest['camera'];msg.width=c['width'];msg.height=c['height']
        K=np.asarray(c['K'],float);msg.k=K.reshape(-1).tolist();msg.d=[0.]*5
        msg.distortion_model='plumb_bob';msg.r=np.eye(3).reshape(-1).tolist()
        # Gazebo camera sensors publish their own local projection matrix.
        msg.p=np.column_stack([K,np.zeros(3)]).reshape(-1).tolist()
        return msg
    counts=Counter();lags=[];start=None
    try:
        deadline=time.monotonic()+args.start_delay_s
        while time.monotonic()<deadline:
            publish_clock(events[0][0]);rclpy.spin_once(node,timeout_sec=.05)
        start=time.monotonic();origin=events[0][0]
        for t,kind,row in events:
            target=start+(t-origin)/1e9/args.rate
            while time.monotonic()<target:
                rclpy.spin_once(node,timeout_sec=min(.01,max(0,target-time.monotonic())))
            lags.append(max(0,time.monotonic()-target));publish_clock(t)
            if kind=='stereo':
                for side,key in [('left','image_path'),('right','right_image_path')]:
                    info=camera_info(t,side);publishers[side+'_info'].publish(info)
                    with PILImage.open(root/row[key]) as im: rgb=np.asarray(im.convert('RGB')).copy()
                    msg=Image();msg.header=info.header;msg.width=info.width;msg.height=info.height
                    msg.encoding='rgb8';msg.is_bigendian=0;msg.step=msg.width*3;msg.data=rgb.tobytes()
                    publishers[side].publish(msg)
            else: publishers[kind].publish(sensor_message(kind,row,Imu if kind=='imu' else JointState))
            counts[kind]+=1;rclpy.spin_once(node,timeout_sec=0.)
        write_json(args.receipt,dict(receipt,status='RECORDED_SENSOR_REPLAY_COMPLETE',published=dict(counts),domain_id=args.domain_id,
            requested_rate=args.rate,elapsed_wall_s=time.monotonic()-start,
            scheduling_lag_s=dict(p50=float(np.median(lags)),p95=float(np.percentile(lags,95)),maximum=max(lags)),
            published_GT=False,published_actuator_commands=False,live_Gazebo_execution=False,
            online_localization_success='requires separate estimator output evaluation'))
    except BaseException as exc:
        write_json(args.receipt,dict(receipt,status='REPLAY_INTERRUPTED_NOT_COMPLETE',error=repr(exc),published=dict(counts)))
        raise
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__': main()
