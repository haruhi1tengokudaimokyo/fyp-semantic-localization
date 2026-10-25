"""Frozen-backbone fine-tuning of official heads. Requires building-disjoint manifest.
For explicit --smoke only, reuse a real public sample for an overfit engineering check.
"""
from common import *
import argparse,time,csv
import torch.nn.functional as F

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=20); p.add_argument('--smoke',action='store_true'); p.add_argument('--accumulate',type=int,default=1); args=p.parse_args()
    if args.steps<1 or args.accumulate<1: raise ValueError('steps and accumulate must be positive')
    manifest=json.loads(Path(args.manifest).read_text()); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        train=manifest[:1] if isinstance(manifest,list) else manifest['engineering_only'][:1]; val=[]
    else:
        train=manifest['train']; val=manifest['val']; test=manifest['test']
        groups=[{read_sample(x)[0]['building_id'] for x in paths} for paths in [train,val,test]]
        if not all(groups) or any(groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)]): raise ValueError('Require nonempty building-disjoint train/val/test; no random image split.')
    torch.manual_seed(42); device='cuda' if torch.cuda.is_available() else 'cpu'
    depth,sem=models(); depth.to(device); sem.to(device)
    for model,backbone in [(depth,depth.depth_feature.resnet),(sem,sem.semantic_feature.resnet)]:
        model.train()
        for v in backbone.parameters(): v.requires_grad=False
        backbone.eval() # Freeze BatchNorm statistics too.
    params=[v for model in [depth,sem] for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4); scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda',init_scale=128.)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    log=open(out/'metrics.csv','w',buffering=1); writer=csv.DictWriter(log,fieldnames=['step','loss','depth_l1','shape_loss','semantic_ce','peak_mib','peak_reserved_mib','amp_scale','optimizer_updates','seconds']); writer.writeheader()
    write_json(out/'status.json',{'state':'running','engineering_smoke':args.smoke,'steps':args.steps,'validation_available':bool(val),'train_count':len(train),'device':device})
    def losses(path):
        meta,img=read_sample(path); img=img.to(device); mask=torch.ones((1,360,640),device=device)
        d,_,_=depth(img,mask); s,_,_=sem(img,mask)
        target=torch.tensor(meta['depth_axial_m'],device=device).unsqueeze(0)
        labels=torch.tensor(meta['semantic'],device=device).unsqueeze(0)
        l1=F.l1_loss(d,target); shape=20*(1-F.cosine_similarity(d,target).mean()); ce=F.cross_entropy(s.permute(0,2,1),labels)
        # Room labels are unknown in sample adapter; do not fabricate supervision.
        room_loss=0.
        if meta.get('room_label_available'):
            _,room,_=sem(img,mask); room_loss=F.cross_entropy(room,torch.tensor([meta['room_label']],device=device))
        return l1+shape+ce+room_loss,l1,shape,ce
    optimizer_updates=0
    start=time.perf_counter(); opt.zero_grad(set_to_none=True)
    for i in range(args.steps):
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            loss,l1,shape,ce=losses(train[i%len(train)])
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        scaler.scale(loss/args.accumulate).backward()
        if (i+1)%args.accumulate==0 or i==args.steps-1:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,1); old_scale=scaler.get_scale(); scaler.step(opt); scaler.update(); optimizer_updates+=int(scaler.get_scale()>=old_scale); opt.zero_grad(set_to_none=True)
        row=dict(step=i+1,amp_scale=scaler.get_scale(),optimizer_updates=optimizer_updates,loss=float(loss.detach()),depth_l1=float(l1.detach()),shape_loss=float(shape.detach()),semantic_ce=float(ce.detach()),peak_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0,seconds=time.perf_counter()-start)
        writer.writerow(row); print(json.dumps(row),flush=True); write_json(out/'status.json',dict(state='running',engineering_smoke=args.smoke,validation_available=bool(val),**row))
    depth.eval(); sem.eval(); validation=[]
    with torch.inference_mode():
        for path in val[:8]: validation.append(float(losses(path)[0]))
    torch.save({'depth':depth.cpu().state_dict(),'semantic':sem.cpu().state_dict(),'engineering_smoke':args.smoke,'validation_loss':validation,'base':'official zind weights','steps':args.steps},out/'heads_finetuned.pt')
    write_json(out/'status.json',dict(state='complete',engineering_smoke=args.smoke,validation_available=bool(val),validation_loss=validation,**row)); log.close()
if __name__=='__main__': main()
