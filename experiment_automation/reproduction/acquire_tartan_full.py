"""Complete trajectories from the publisher's official Hugging Face mirror."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from blocked_data import selected_zip,atomic

def main(environment,sequence,out):
    out=Path(out);prefix=f'{environment}/Easy/{sequence}/'
    receipts=[]
    for camera in ['left','right']:
        url=f'https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{environment}/Easy/image_{camera}.zip'
        folder=out/f'archive_{camera}'
        receipts+=selected_zip(url,folder,lambda n:n.startswith(prefix) and (f'image_{camera}/' in n or n.endswith('pose_left.txt')))
        source=folder/prefix
        target=out/'sequence';target.mkdir(exist_ok=True)
        dst=target/('image_'+camera)
        if not dst.exists():dst.symlink_to(source/('image_'+camera),target_is_directory=True)
        pose=source/'pose_left.txt'
        if camera=='left' and not (target/'pose_left.txt').exists():(target/'pose_left.txt').symlink_to(pose)
    left=sorted((target/'image_left').glob('*.png'));right=sorted((target/'image_right').glob('*.png'))
    L=[p.name.split('_')[0] for p in left];R=[p.name.split('_')[0] for p in right]
    assert L==R and len(L)>200
    assert L==[f'{i:06d}' for i in range(len(L))], 'Full original frame identities must be contiguous'
    poses=np.loadtxt(target/'pose_left.txt');assert poses.shape==(len(L),7) and np.isfinite(poses).all()
    sizes=set()
    for p in left+right:
        with Image.open(p) as im: im.load();sizes.add(im.size)
    assert sizes=={(640,480)}
    atomic(out/'acceptance.json',dict(status='accepted',dataset='TartanAir v1',environment=environment,difficulty='Easy',sequence=sequence,frames=len(L),first_frame=L[0],last_frame=L[-1],full_source_sequence=True,camera_size=[640,480],K=[[320,0,320],[0,320,240],[0,0,1]],baseline_m=.25,pose_convention='Publisher NED pose_left, xyz/quaternion xyzw',source='theairlabcmu/tartanair, linked by publisher castacks/tartanair_tools',CRC_SHA256_members_verified=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('environment');p.add_argument('sequence');p.add_argument('output');a=p.parse_args();main(a.environment,a.sequence,a.output)
