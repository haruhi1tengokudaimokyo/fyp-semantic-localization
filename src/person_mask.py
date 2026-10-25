"""Optional torchvision LR-ASPP MobileNetV3 pretrained person segmentation.
No FYP data training. Download only on explicit invocation; TORCH_HOME stays on /data.
"""
from common import *

def person_ray_confidence(rgb_batch,device='cuda',threshold=.5):
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights
    torch.hub.set_dir(str(ROOT/'models/torchhub'))
    weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model=lraspp_mobilenet_v3_large(weights=weights).to(device).eval()
    person=weights.meta['categories'].index('person')
    with torch.inference_mode():
        logits=model(weights.transforms()(rgb_batch.to(device)))['out']
        prob=logits.softmax(1)[:,person:person+1]
        prob=torch.nn.functional.interpolate(prob,size=rgb_batch.shape[-2:],mode='bilinear',align_corners=False)
        mask=prob>threshold
        # Expand spatial mask to allow uncertain boundaries. Per-column coverage is a conservative heuristic.
        mask=torch.nn.functional.max_pool2d(mask.float(),kernel_size=21,stride=1,padding=10)
        coverage=torch.nn.functional.adaptive_avg_pool2d(mask,(1,40)).flatten().cpu().numpy()
    return np.clip(1-coverage,0,1),mask[0,0].cpu().numpy()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    _,img=read_sample(args.sample); conf,mask=person_ray_confidence(img,'cuda' if torch.cuda.is_available() else 'cpu')
    from PIL import Image
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True); Image.fromarray((mask*255).astype('uint8')).save(out/'person_mask.png');write_json(out/'confidence.json',{'ray_confidence':conf.tolist(),'person_pixel_fraction':float(mask.mean()),'detector':'torchvision LRASPP MobileNetV3 Large pretrained','note':'Detector not validated on indoor robot people or occlusion benchmark.'})
