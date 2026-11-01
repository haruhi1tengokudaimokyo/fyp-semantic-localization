"""Actual RGB-D motion for SG2Loc; no query pose/instance annotation is opened."""
import argparse,sys
from pathlib import Path
import numpy as np,cv2
from blocked_data import atomic
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'research3d/scripts'))
from run_tum_rgbd_vo import estimate
def main(data,out):
    data,out=Path(data),Path(out);out.mkdir(parents=True,exist_ok=True)
    K=np.load(data/'K.npy');cv2.setNumThreads(2);cv2.setRNGSeed(2089)
    poses=[];lost=[];T=np.eye(4);previous=None;depth=None
    for i,p in enumerate(sorted((data/'query').glob('*.png'))):
        gray=cv2.imread(str(p),0);ok=i==0
        if previous is not None:
            delta,ok,n=estimate(previous,gray,depth,K)
            if ok:T=T@delta
        poses.append(T.copy());lost.append(not ok)
        # Depth is a real simulated sensor observation, not scene ground truth poses.
        depth=np.load(data/'query'/(p.stem+'.npz'))['depth'];previous=gray
    assert len(poses)==600
    # A known level camera mounting fixes roll/pitch, without its unknown heading
    # or position. The PF still samples map-wide translation and yaw.
    mount=np.array([[1.,0,0,0],[0,0,1.,0],[0,-1.,0,0],[0,0,0,1.]])
    np.savez_compressed(out/'measured_motion.npz',T_local_camera=mount@np.stack(poses),lost=lost)
    atomic(out/'motion.json',dict(status='accepted',frames=600,GT_online=False,instance_labels_online=False,method='Measured GFTT/LK RGB-D PnP odometry; original SG2Loc transition/filter unchanged',lost_frames=int(np.sum(lost)),gravity='Declared level camera mounting, no per-frame GT orientation',world_origin_unknown=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args();main(a.data,a.output)
