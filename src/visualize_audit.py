"""Evidence-based diagnostic figures, never synthetic localization results or invented tracks."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image
from common import ROOT
from localization.geometry import pose_error

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.out);plots=out/'plots';plots.mkdir(parents=True,exist_ok=True);cases=out/'cases';cases.mkdir(exist_ok=True)
 s=json.loads((run/'summary.json').read_text());cm=np.array(s['perception']['confusion_matrix_gt_rows_pred_columns'])
 fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');im=ax[0].imshow(cm,cmap='Blues');ax[0].set_xticks(range(4),['wall','window','door','unknown']);ax[0].set_yticks(range(4),['wall','window','door','unknown']);ax[0].set(xlabel='prediction',ylabel='GT',title='Argmax confusion: 1,280 rays / 32 images')
 for i in range(4):
  for j in range(4):ax[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>500 else 'black')
 categories=s['perception']['classes'][:3];x=np.arange(3);ax[1].bar(x-.15,[v['precision'] for v in categories],.3,label='precision');ax[1].bar(x+.15,[v['recall'] for v in categories],.3,label='recall');ax[1].set_xticks(x,['wall','window','door']);ax[1].set(ylim=(0,1),title='Class metrics; unknown has no GT support');ax[1].legend();fig.savefig(plots/'semantic_diagnosis.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');conditions=['A_gt_gt','B_pred_gt','C_gt_pred','D_pred_pred'];x=np.arange(4)
 for ax,profile in zip(axes,['official_reproduction','our_integration']):
  for delta,stage in [(-.17,'coarse'),(.17,'fine')]:ax.bar(x+delta,[s['cases'][profile+'/'+c+'/combined'][stage]['recall']['1m 30 deg']*100 for c in conditions],.34,label=stage)
  ax.set_xticks(x,['A: GT/GT','B: Pred/GT','C: GT/Pred','D: Pred/Pred']);ax.set(ylim=(0,100),ylabel='Recall @ 1m, 30deg (%)',title=profile+' / no room prior');ax.tick_params(axis='x',labelrotation=15);ax.legend()
 fig.savefig(plots/'diagnostic_matrix.png',dpi=160);plt.close(fig)
 sensitivity=json.loads((ROOT/'runs/audit_v2/perception_sensitivity/summary.json').read_text())['conditions'];fractions=[float(k) for k in sensitivity]
 fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
 for axis,key,title in zip(ax,['mean_axial_depth_mae_m','mean_semantic_entropy','accuracy'],['Axial depth MAE (m)','Semantic entropy (nats)','Semantic accuracy']):
  values=[v['semantic']['accuracy'] if key=='accuracy' else v[key] for v in sensitivity.values()];axis.plot(np.array(fractions)*100,values,'o-');axis.set(xlabel='gray rectangle coverage (%)',title=title)
 fig.suptitle('Controlled appearance sensitivity; no oracle mask fed to the algorithm; NOT real people');fig.savefig(plots/'occlusion_sensitivity.png',dpi=160);plt.close(fig)
 # Every image is shown in original index order, including failures.
 palette=ListedColormap(['#454545','#3d70d8','#dd4444','#dc8ad3'])
 for file in sorted((run/'per_image').glob('*.json')):
  item=json.loads(file.read_text());idx=int(file.stem);meta=json.loads((Path(item['sample'])/'sample.json').read_text());gt=np.array(item['ground_truth_pose']);heat=np.load(file.with_name(file.stem+'_heatmaps.npz'));map_image=np.asarray(Image.open(meta['map']));fig=plt.figure(figsize=(17,13),layout='constrained');grid=fig.add_gridspec(4,4)
  rgb_ax=fig.add_subplot(grid[0,:2]);rgb_ax.imshow(Image.open(Path(item['sample'])/'rgb.png'));rgb_ax.set_title(f'Actual query {idx:03d} | no training in this audit');rgb_ax.axis('off')
  map_ax=fig.add_subplot(grid[0,2:]);map_ax.imshow(map_image);r=item['profiles']['our_integration']['D_pred_pred']['combined'];pred=np.array(r['estimated_pose']);map_ax.scatter(gt[0]*100,gt[1]*100,c='lime',label='GT',s=45);map_ax.scatter(pred[0]*100,pred[1]*100,c='magenta',label='estimate',s=45)
  for j,c in enumerate(r['candidates']):v=np.array(c['pose']);map_ax.scatter(v[0]*100,v[1]*100,c='orange',s=12);map_ax.text(v[0]*100,v[1]*100,str(j+1),fontsize=7)
  map_ax.legend(fontsize=8);map_ax.set_title('Fixed semantic map / distinct candidates');map_ax.axis('off')
  depth_ax=fig.add_subplot(grid[1,:2]);depth_ax.plot(item['ground_truth_depth_axial'],label='GT axial');depth_ax.plot(item['ground_truth_depth_radial'],label='GT radial',alpha=.7);depth_ax.plot(item['predicted_depth'],label='network output');depth_ax.set(xlabel='image-column ray',ylabel='m',title='Depth conventions are displayed, not silently changed');depth_ax.legend(fontsize=7)
  sem_ax=fig.add_subplot(grid[1,2]);prob=np.array(item['predicted_semantic_probability']);sam=np.array(item['profiles']['official_reproduction']['D_pred_pred']['combined']['candidates'][0]['pose'])
  sem_ax.imshow(np.stack([item['ground_truth_semantic'],prob.argmax(-1)]),aspect='auto',cmap=palette,vmin=0,vmax=3);sem_ax.set_yticks([0,1],['GT','argmax']);sem_ax.set(xlabel='ray',title='gray wall / blue window / red door')
  prob_ax=fig.add_subplot(grid[1,3]);prob_ax.imshow(prob.T,aspect='auto',vmin=0,vmax=1,cmap='viridis');prob_ax.set_yticks(range(4),['wall','window','door','unknown']);prob_ax.set(xlabel='ray',title='Predicted semantic probability')
  for row,profile in [(2,'our_integration'),(3,'official_reproduction')]:
   for col,condition in enumerate(conditions):
    axis=fig.add_subplot(grid[row,col]);axis.imshow(heat[f'{profile}_{condition}_combined_maxscore'],cmap='magma');result=item['profiles'][profile][condition]['combined'];pose=np.array(result['estimated_pose']);coarse=np.array(result['coarse_pose']);xy,yaw=pose_error(pose,gt)
    axis.scatter(gt[0]*10,gt[1]*10,c='lime',s=14);axis.scatter(coarse[0]*10,coarse[1]*10,c='cyan',s=10);axis.scatter(pose[0]*10,pose[1]*10,c='magenta',s=12);axis.set_title(f'{profile.split("_")[0]} {condition[:1]}: {xy:.2f}m / {yaw:.0f}deg',fontsize=9);axis.axis('off')
  fig.suptitle('Raw score heatmaps are uncalibrated; green GT / cyan coarse / magenta fine. Single building diagnostic only.',fontsize=11);fig.savefig(cases/f'{idx:03d}.png',dpi=130);plt.close(fig)
  print('rendered',idx,flush=True)
 # Existing bag trajectories are labeled as recorded signals, never model predictions.
 pre=ROOT/'runs/audit_v2/rosbag_preflight';bag=json.loads((pre/'preflight.json').read_text());fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');ax[0].imshow(Image.open(pre/'first_rgb.png'));ax[0].set_title('Recorded Gazebo RGB / 320x240 / 60deg');ax[0].axis('off')
 world=ET.parse('/data/FYP_Indoor_Localization/ros2_ws/src/fyp_sim/worlds/indoor.sdf')
 for model in world.findall('.//world/model'):
  if model.get('name') in ['floor','fyp_robot']:continue
  pose=[float(x) for x in model.findtext('pose').split()];box=model.find('.//collision/geometry/box/size')
  if box is not None:
   dims=[float(x) for x in box.text.split()];ax[1].add_patch(Rectangle((pose[0]-dims[0]/2,pose[1]-dims[1]/2),dims[0],dims[1],color='#555' if model.get('name') in ['north','south','east','west'] else '#bfbfbf'))
 gt=np.array([[v['x'],v['y']] for v in bag['ground_truth']]);od=np.array([[v['x'],v['y']] for v in bag['odom']]);ax[1].plot(gt[:,0],gt[:,1],c='tab:green',label='recorded GT (world)');ax[1].set(xlim=(-5.5,5.5),ylim=(-4.5,4.5),title='Known SDF geometry + recorded GT');ax[1].set_aspect('equal');ax[1].legend(fontsize=8)
 ax[2].plot(od[:,0],od[:,1],c='tab:orange');ax[2].set(title='Recorded odometry in odom frame',xlabel='x (m)',ylabel='y (m)');ax[2].set_aspect('equal');fig.suptitle('Preflight only: these are NOT trajectories predicted by SemRayLoc; no temporal estimator was run');fig.savefig(plots/'rosbag_preflight.png',dpi=150);plt.close(fig)
 html='''<!doctype html><meta charset="utf-8"><title>P0/P1 actual diagnostic gallery</title><style>body{background:#172030;color:#edf3ff;font:17px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%;background:white}select{font-size:20px;padding:8px}a{color:#a9d8ff}</style><h1>P0/P1 actual diagnostic gallery</h1><p>32 related views, one public building. No held-out baseline claim. Every view is shown, including failures. No model training occurred in this audit.</p><label>Query index <select id="query">'''+''.join(f'<option>{i:03d}</option>' for i in range(32))+'''</select></label><img id="case" src="cases/000.png"><script>document.getElementById('query').onchange=e=>document.getElementById('case').src='cases/'+e.target.value+'.png';</script>'''
 (out/'gallery.html').write_text(html)
if __name__=='__main__':main()
