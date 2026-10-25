"""Thin integration layer. All ray networks are imported from pinned SemRayLoc."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ.get('FYP_ROOT','/data/fyp-semrayloc'))
REPO=Path(os.environ.get('SEMRAYLOC_REPO',str(ROOT/'repos/SemRayLoc')))
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'data_utils/zind'))
torch.set_num_threads(4)

def models(dataset='zind', weights=True):
    # Checkpoint already contains backbone; avoid an unnecessary ImageNet download.
    import modules.depth.depth_net as dn
    import modules.semantic.semantic_and_room_type_with_cls as sn
    from torchvision.models import resnet50
    dn.resnet50=lambda **kw: resnet50(weights=None,replace_stride_with_dilation=kw['replace_stride_with_dilation'])
    sn.resnet50=dn.resnet50
    depth=dn.depth_net(); semantic=sn.semantic_net(4,14 if dataset=='zind' else 16)
    if weights:
        for model,name,prefix in [(depth,'depth','encoder.'),(semantic,'semantic','model.')]:
            path=ROOT/'models'/dataset/f'final_{name}.ckpt'
            state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
            state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            model.load_state_dict(state,strict=True)
    if os.environ.get('FYP_FINETUNED'):
        state=torch.load(os.environ['FYP_FINETUNED'],map_location='cpu',weights_only=True)
        depth.load_state_dict(state['depth'],strict=True); semantic.load_state_dict(state['semantic'],strict=True)
    return depth,semantic

def read_sample(path):
    from PIL import Image
    p=Path(path); meta=json.loads((p/'sample.json').read_text())
    image=np.asarray(Image.open(p/'rgb.png').convert('RGB'),dtype=np.float32)/255
    return meta,torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0)

def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
