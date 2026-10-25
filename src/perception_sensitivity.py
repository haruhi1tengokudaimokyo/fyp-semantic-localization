"""Controlled gray-rectangle sensitivity ONLY; not real people or an algorithmic improvement."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,read_sample,write_json
from localization.models.observation import ObservationModel
from localization.evaluation.metrics import classification_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 paths=json.loads((ROOT/'data/zind_public_sample/manifest.json').read_text());model=ObservationModel();summary={};all_rows=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temp:
  for fraction in [0.,.1,.25,.4,.6]:
   gt=[];pred=[];prob=[];errors=[];entropy=[]
   for i,path in enumerate(paths):
    meta,_=read_sample(path);image=np.array(Image.open(Path(path)/'rgb.png').convert('RGB'));width=int(640*fraction);left=(640-width)//2
    if width:image[:,left:left+width]=64
    query=Path(temp)/'query.png';Image.fromarray(image).save(query)
    # No oracle mask is passed to model or matcher.
    ob=model.predict(query);labels=ob.labels();gt.extend(meta['semantic']);pred.extend(labels);prob.extend(ob.semantic_probability);mae=float(abs(ob.depth-np.array(meta['depth_axial_m'])).mean());errors.append(mae)
    ent=float(-(ob.semantic_probability*np.log(np.maximum(ob.semantic_probability,1e-12))).sum(-1).mean());entropy.append(ent)
    all_rows.append({'sample_index':i,'occlusion_fraction':fraction,'gray_rectangle_augmentation':True,'oracle_mask_passed_to_algorithm':False,'depth_mae_axial_m':mae,'semantic_entropy':ent,'semantic_prediction':labels.tolist(),'predicted_depth':ob.depth.tolist()})
   summary[str(fraction)]={'images':len(paths),'mean_axial_depth_mae_m':float(np.mean(errors)),'mean_semantic_entropy':float(np.mean(entropy)),'semantic':classification_metrics(gt,pred,prob)};print('completed occlusion',fraction,flush=True)
 write_json(out/'summary.json',{'scope':'network sensitivity diagnostic, same single building; not dynamic-person benchmark','conditions':summary});write_json(out/'per_image.json',all_rows)
if __name__=='__main__':main()
