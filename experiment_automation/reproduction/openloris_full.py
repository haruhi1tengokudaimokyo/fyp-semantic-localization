"""Full publisher ROS bags, not fixed-duration cuts; audited against source records."""
import argparse,json,tarfile,hashlib,shutil
from pathlib import Path
from blocked_data import RemoteZip,atomic

def acquire(name,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip('https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/rosbag/cafe1-1_2-rosbag.tar')
    offset=0;member=None
    while offset<remote.size:
        remote.seek(offset);header=remote.read(512)
        if not header.strip(b'\0'):break
        info=tarfile.TarInfo.frombuf(header,'utf-8','surrogateescape')
        if info.name==name+'.bag':member=info;break
        offset+=512+((info.size+511)//512)*512
    assert member is not None
    target=out/(name+'.bag');part=out/(name+'.bag.part');identity=out/'partial_identity.json'
    binding=dict(url=remote.url,etag=remote.etag,member=member.name,bytes=member.size,offset_data=offset+512)
    if part.exists():assert json.loads(identity.read_text())==binding
    else:atomic(identity,binding)
    count=part.stat().st_size if part.exists() else 0
    assert shutil.disk_usage(out).free-(member.size-count)>30*1024**3
    remote.seek(offset+512+count)
    with part.open('ab') as f:
        while count<member.size:
            assert shutil.disk_usage(out).free>30*1024**3
            block=remote.read(min(8*1024**2,member.size-count));assert block
            f.write(block);count+=len(block)
            if count%(128*1024**2)==0:print('BAG_BYTES',name,count,member.size,flush=True)
    assert count==member.size
    h=hashlib.sha256()
    with part.open('rb') as f:
        while b:=f.read(4*1024**2):h.update(b)
    from rosbags.rosbag1 import Reader
    with Reader(part) as r:
        duration=r.duration/1e9;topics={c.topic:c.msgcount for c in r.connections}
    part.replace(target)
    atomic(out/'acceptance.json',dict(status='accepted',source_binding=binding,sha256=h.hexdigest(),bytes=member.size,duration_s=duration,topics=topics,full_source_bag=True,bag=str(target),not_a_fixed_duration_excerpt=True))


def coverage(bag,prepared,output):
    from rosbags.rosbag1 import Reader
    bag=Path(bag);prepared=Path(prepared)
    with Reader(bag) as r:
        topics={c.topic:c.msgcount for c in r.connections};duration=r.duration/1e9
    audit=json.loads((prepared/'audit.json').read_text())
    for topic in ['/t265/fisheye1/image_raw','/t265/fisheye2/image_raw','/odom']:
        assert topics[topic]==audit['online_topics'][topic], f'Full-source message loss for {topic}'
    import numpy as np
    times=np.load(prepared/'imu_calibrated.npz')['timestamp_ns']
    atomic(output,dict(status='accepted',full_source_bag=True,all_camera_messages_preserved=True,source_record_duration_s=duration,merged_IMU_header_duration_s=float((times[-1]-times[0])/1e9),GT_online=False,LiDAR_GPS_online=False,online_topics=audit['online_topics'],pure_wheel=False,odom_source='Publisher Segway vendor encoder+IMU fusion; not raw wheel'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['acquire','coverage']);p.add_argument('arguments',nargs='+');a=p.parse_args()
    if a.phase=='acquire':acquire(*a.arguments)
    else:coverage(*a.arguments)
