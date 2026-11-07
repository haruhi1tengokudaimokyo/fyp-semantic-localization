"""Predeclared sensor blackout for failure testing, with no pose/GT input.

Both stereo images keep their original capture timestamps. This simulates a
covered/failed camera at the image interface; it is not a physical occluder or
a claim about natural pedestrian occlusion. The modified RGB is recorded and
consumed by every downstream algorithm, so replay sees the same observations.
"""
import argparse
from array import array
import json
from pathlib import Path


def blackout_active(timestamp_ns, origin_ns, start_s, end_s):
    """Use sensor integer time; float subtraction can exclude the start frame."""
    return round(start_s*1_000_000_000) <= timestamp_ns-origin_ns < round(end_s*1_000_000_000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-s",type=float,required=True)
    p.add_argument("--end-s",type=float,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not 0<=a.start_s<a.end_s:p.error("increasing nonnegative fault interval required")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    rclpy.init()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    records=a.output.open("w",buffering=1)
    class Fault(Node):
        def __init__(self):
            super().__init__("declared_stereo_camera_fault")
            self.origin=None;self.origin_ns=None;self.counts={"left":0,"right":0};self.blackouts={"left":0,"right":0}
            self.pubs={side:self.create_publisher(Image,f"/fault_camera/{side}/image",qos_profile_sensor_data)
                       for side in ("left","right")}
            self.create_subscription(Image,"/robot/camera/image",lambda m:self.callback(m,"left"),qos_profile_sensor_data)
            self.create_subscription(Image,"/robot/camera_right/image",lambda m:self.callback(m,"right"),qos_profile_sensor_data)
        def callback(self,msg,side):
            ns=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            stamp=ns/1e9
            if self.origin_ns is None:self.origin_ns=ns;self.origin=stamp
            active=blackout_active(ns,self.origin_ns,a.start_s,a.end_s)
            if active:
                msg.data=array('B',bytes(len(msg.data)));self.blackouts[side]+=1
            self.counts[side]+=1
            self.pubs[side].publish(msg)
            records.write(json.dumps(dict(timestamp_s=stamp,timestamp_ns=ns,side=side,blackout=active,
                                         fault="synthetic_camera_blackout",GT_read=False))+"\n")
    node=Fault()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        records.close()
        a.output.with_suffix(".summary.json").write_text(json.dumps(dict(origin_timestamp_s=node.origin,
            origin_timestamp_ns=node.origin_ns,timing_policy="integer sensor nanoseconds; start inclusive/end exclusive",
            interval_s=[a.start_s,a.end_s],counts=node.counts,blackouts=node.blackouts,
            scope="declared image-interface camera failure; not physical or natural occlusion",GT_read=False),indent=2)+"\n")
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
