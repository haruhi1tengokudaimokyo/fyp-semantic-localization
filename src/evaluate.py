from common import *
from matching import atlas,match
from evaluation.result_handler import calculate_recalls
import time,csv,argparse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=8); args=p.parse_args()
    paths=json.loads(Path(args.manifest).read_text())[:args.limit]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'; depth,semantic=models(); depth.to(device).eval(); semantic.to(device).eval()
    torch.manual_seed(42); rows=[]
    with torch.inference_mode():
        for ix,path in enumerate(paths):
            meta,img=read_sample(path); a=atlas(meta); base=img.to(device)
            # Four separate measurements: clean, occluded, mask only, mask + JSD.
            for fraction,mode in [(0.,'official'),(.25,'official'),(.25,'mask_only'),(.25,'weighted_jsd')]:
                x=base.clone(); confidence=np.ones(40)
                if fraction:
                    x[:,:,:,240:400]=.25; confidence[15:25]=0
                mask=torch.ones((1,360,640),device=device)
                if device=='cuda': torch.cuda.synchronize()
                start=time.perf_counter(); d,_,_=depth(x,mask); s,_,_=semantic(x,mask)
                if device=='cuda': torch.cuda.synchronize()
                net=1000*(time.perf_counter()-start)
                dr=d[0].cpu().numpy(); sp=s.softmax(-1)[0].cpu().numpy(); hard=sp.argmax(-1)
                start=time.perf_counter()
                if mode=='mask_only':
                    candidates,_=match(a,dr,hard,confidence,sp,'mask_only')
                else: candidates,_=match(a,dr,hard,confidence,sp,mode)
                ms=1000*(time.perf_counter()-start); pred=np.array(candidates[0]['pose']); gt=np.array(meta['pose'])
                row={'sample':path,'condition':f'{fraction:g}_{mode}','error_m':float(np.linalg.norm(pred[:2]-gt[:2])),'error_deg':float(abs((pred[2]-gt[2]+np.pi)%(2*np.pi)-np.pi)*180/np.pi),'network_ms':net,'matching_ms':ms}; rows.append(row)
            print(f'completed {ix+1}/{len(paths)}',flush=True)
    summary={'scope':'single public building, overlapping views; not held-out benchmark; pretrained membership unknown','count_images':len(paths),'building_count':1,'grid_m':.1,'refinement':False,'room_prior':False,'occlusion':'25% central gray rectangle, oracle mask; not real people','conditions':{}}
    for condition in sorted({r['condition'] for r in rows}):
        v=[r for r in rows if r['condition']==condition]; e=np.array([r['error_m'] for r in v]); angle=np.array([r['error_deg'] for r in v])
        summary['conditions'][condition]={'n':len(v),'median_error_m':float(np.median(e)),'median_error_deg':float(np.median(angle)),'mean_network_ms':float(np.mean([r['network_ms'] for r in v])),'mean_matching_ms':float(np.mean([r['matching_ms'] for r in v])),'recall':calculate_recalls(e,angle)}
    write_json(out/'summary.json',summary)
    with open(out/'per_image.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
