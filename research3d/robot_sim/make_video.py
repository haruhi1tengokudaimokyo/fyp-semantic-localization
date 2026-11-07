"""Offline GIF/trajectory view of actual RGB and same-query model outputs.

Ground truth is used only by this independent visualization, including one
explicit first-valid-pose gauge transform for local MAC-VO diagnostics. Missing
outputs remain gaps; no pose interpolation or nearest-query substitution.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from research3d.fusion.contracts import pose,read_jsonl,sha256_file,write_json


def indexed(path):
    if path is None:return {}
    result={}
    for row in read_jsonl(path):
        if row['query_id'] in result:raise ValueError('duplicate output query id')
        result[row['query_id']]=row
    return result


def first_pose_gauge(frames,truth,macvo):
    for row in frames:
        q=row['query_id'];m=macvo.get(q)
        if m and m['status']=='estimated':
            return pose(truth[q]['T_map_camera'])@np.linalg.inv(pose(m['T_odom_camera'])),q
    return None,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--ace',type=Path)
    p.add_argument('--spvloc',type=Path)
    p.add_argument('--macvo',type=Path)
    p.add_argument('--fusion',type=Path)
    p.add_argument('--fusion-timing',choices=['offline_accuracy','measured_service_fifo'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=5)
    a=p.parse_args()
    if a.stride<1:p.error('stride must be positive')
    if a.fusion is not None and not a.fusion_timing:p.error('declare fusion timing mode')
    if a.output.exists():raise FileExistsError('refusing to overwrite visualization')
    dataset=a.dataset.resolve(strict=True);a.output.mkdir(parents=True)
    manifest=json.loads((dataset/'map_manifest.json').read_text())
    frames=read_jsonl(dataset/'query/frames.jsonl');truth=indexed(dataset/'offline_gt/query_poses.jsonl')
    if len(frames)<3:raise ValueError('at least three actual trajectory frames required')
    ace,spv,mac,fusion=[indexed(path) for path in (a.ace,a.spvloc,a.macvo,a.fusion)]
    input_index={r['query_id']:r for r in frames}
    if set(input_index)!=set(truth):raise ValueError('GT does not cover exact frozen query inventory')
    for name,items in [('ACE',ace),('SPVLoc',spv),('MACVO',mac),('fusion',fusion)]:
        for q,row in items.items():
            if q not in input_index or row['map_id']!=manifest['map_id']:raise ValueError(name+' output/query map mismatch')
            frame=input_index[q]
            t=row.get('timestamp_ns',round(row.get('timestamp_s',0)*1e9))
            if abs(t-frame['timestamp_ns'])>1000:raise ValueError(name+' output/image timestamp mismatch')
            if 'image_sha256' in row and row['image_sha256']!=frame['image_sha256']:raise ValueError(name+' used different image')
    gauge,gauge_qid=first_pose_gauge(frames,truth,mac)
    ids=[r['query_id'] for r in frames]
    def points(method):
        values=[]
        for q in ids:
            row=method.get(q);T=None
            if method is ace and row and row['status']=='ok':T=row['T_map_camera_optical']
            elif method is spv and row and row['status']=='ok':T=row['refined_top1_T_map_camera_optical']
            elif method is mac and row and row['status']=='estimated' and gauge is not None:T=gauge@pose(row['T_odom_camera'])
            elif method is fusion and row and row['valid']:T=row['T_map_camera']
            values.append([np.nan,np.nan] if T is None else pose(T)[:2,3])
        return np.asarray(values,float)
    gt=np.asarray([pose(truth[q]['T_map_camera'])[:2,3] for q in ids])
    trajectories=[('ACE',points(ace),'tab:orange','o'),('SPVLoc refined Top1',points(spv),'tab:purple','^'),
                  ('MAC-VO: first-pose GT origin ONLY',points(mac),'tab:blue','.'),
                  ('Fusion cached: '+str(a.fusion_timing),points(fusion),'tab:green','s')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(5,4),dpi=100)
    for obj in manifest.get('objects',[]):
        if obj['category']=='wall':
            x,y,_=obj['center_m'];sx,sy,_=obj['size_m']
            ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,facecolor='.87',edgecolor='.75',linewidth=.4))
    other_candidates=np.asarray([pose(c['T_map_camera_optical'])[:2,3] for row in spv.values()
                                 if row['status']=='ok' for c in row['candidates'][1:3]],float).reshape(-1,2)
    all_xy=np.vstack([gt,other_candidates]+[xy[np.isfinite(xy).all(axis=1)] for _,xy,_,_ in trajectories])
    # Show actual outliers rather than silently clipping them to the building.
    lo=all_xy.min(axis=0)-.7;hi=all_xy.max(axis=0)+.7
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='map x (m)',ylabel='map y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.2)
    gt_line,=ax.plot([],[],color='black',linewidth=1,label='GT: OFFLINE DISPLAY ONLY')
    lines=[];current=[]
    for name,xy,color,marker in trajectories:
        # Sparse absolute fixes are points; no lines through unobserved instants.
        line,=ax.plot([],[],color=color,marker=marker,markersize=2,linewidth=.8 if name.startswith(('MAC','Fusion')) else 0,label=name)
        now,=ax.plot([],[],color=color,marker=marker,markersize=7,linestyle='None')
        lines.append(line);current.append(now)
    topk,=ax.plot([],[],marker='^',color='magenta',linestyle='None',markersize=5,alpha=.5,label='SPVLoc initial ranks 2/3')
    zoom=ax.inset_axes([.53,.02,.45,.42])
    zoom.set(xlim=(gt[:,0].min()-.6,gt[:,0].max()+.6),ylim=(gt[:,1].min()-.6,gt[:,1].max()+.6))
    zoom.set_aspect('equal');zoom.tick_params(labelsize=5);zoom.grid(alpha=.2)
    zoom.set_title('Route detail; all outliers shown in overview',fontsize=5)
    zoom_gt,=zoom.plot([],[],color='black',linewidth=1.5)
    zoom_lines=[]
    for name,_,color,marker in trajectories:
        line,=zoom.plot([],[],color=color,marker=marker,markersize=2,linewidth=.8 if name.startswith(('MAC','Fusion')) else 0,alpha=.6)
        zoom_lines.append(line)
    ax.legend(fontsize=5.4,loc='upper left');fig.tight_layout()
    chosen=list(range(0,len(frames),a.stride))
    if chosen[-1]!=len(frames)-1:chosen.append(len(frames)-1)
    gif=[];durations=[]
    for index,i in enumerate(chosen):
        frame=frames[i];qid=frame['query_id'];gt_line.set_data(gt[:i+1,0],gt[:i+1,1]);zoom_gt.set_data(gt[:i+1,0],gt[:i+1,1])
        text=[]
        for j,(name,xy,color,marker) in enumerate(trajectories):
            lines[j].set_data(xy[:i+1,0],xy[:i+1,1])
            zoom_lines[j].set_data(xy[:i+1,0],xy[:i+1,1])
            valid=bool(np.isfinite(xy[i]).all())
            current[j].set_data([xy[i,0]] if valid else [],[xy[i,1]] if valid else [])
            text.append(name.split(':')[0]+(': output' if valid else ': MISSING'))
        candidate=spv.get(qid)
        fusion_row=fusion.get(qid)
        gate_state=fusion_row.get('status','UNKNOWN') if fusion_row else 'NO OUTPUT RECORD'
        pts=[]
        if candidate and candidate['status']=='ok':
            pts=[pose(c['T_map_camera_optical'])[:2,3] for c in candidate['candidates'][1:3]]
        topk.set_data([p[0] for p in pts],[p[1] for p in pts])
        elapsed=(frame['timestamp_ns']-frames[0]['timestamp_ns'])/1e9
        ax.set_title(f'Camera trajectory / recorded simulation t={elapsed:.1f}s',fontsize=8)
        fig.canvas.draw();plot=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
        path=(dataset/frame['image_path']).resolve(strict=True)
        if not path.is_relative_to(dataset/'query') or sha256_file(path)!=frame['image_sha256']:raise ValueError('RGB changed')
        with Image.open(path) as im:rgb=im.convert('RGB').resize((480,360))
        canvas=Image.new('RGB',(980,470),'white');canvas.paste(rgb,(0,20));canvas.paste(plot,(480,0))
        draw=ImageDraw.Draw(canvas);draw.text((10,3),'ACTUAL GAZEBO LEFT RGB / '+qid,fill='black')
        draw.text((10,390),' | '.join(text),fill='black')
        draw.text((10,410),'OFFLINE visualization: GT never enters online localization. Missing poses remain gaps.',fill='black')
        draw.text((10,430),'MAC-VO receives one GT origin for local diagnostic only; it is NOT global localization.',fill='navy')
        draw.text((10,450),'Fusion gate: '+gate_state+' / '+str(a.fusion_timing)+'; cached replay is NOT real-time deployment evidence.',fill='darkgreen')
        gif.append(canvas.quantize(colors=128))
        next_i=chosen[index+1] if index+1<len(chosen) else i
        duration=(frames[next_i]['timestamp_ns']-frame['timestamp_ns'])/1e6 if next_i!=i else 500
        durations.append(max(20,int(round(duration/10)*10)))
    gif[0].save(a.output/'localization_replay.gif',save_all=True,append_images=gif[1:],duration=durations,loop=0,optimize=False)
    gif[-1].convert('RGB').save(a.output/'final_frame.png')
    fig.savefig(a.output/'trajectory.png',dpi=180);plt.close(fig)
    write_json(a.output/'visualization_receipt.json',dict(status='OFFLINE_VISUALIZATION_COMPLETE',frames=len(frames),gif_frames=len(gif),
        source_duration_s=(frames[-1]['timestamp_ns']-frames[0]['timestamp_ns'])/1e9,
        displayed_duration_s=sum(durations)/1000,GT_scope='offline display/evaluation only',
        MACVO_alignment='one first-valid-camera pose origin, no full-trajectory fit',MACVO_anchor_query_id=gauge_qid,
        interpolation=False,missing_outputs='gaps and explicit MISSING status',fusion_timing=a.fusion_timing,real_time_claim=False,
        map_id=manifest['map_id'],gif_bytes=(a.output/'localization_replay.gif').stat().st_size,
        input_sha256={str(p.resolve()):sha256_file(p) for p in [dataset/'query/frames.jsonl',dataset/'offline_gt/query_poses.jsonl',a.ace,a.spvloc,a.macvo,a.fusion] if p is not None}))
    print(json.dumps(dict(output=str(a.output),frames=len(gif))))


if __name__=='__main__':main()
