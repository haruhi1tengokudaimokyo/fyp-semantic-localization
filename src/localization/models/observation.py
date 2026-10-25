"""Online RGB-only observation interface. Never receives pose, map GT rays or oracle masks."""
from dataclasses import dataclass
import hashlib,time
import numpy as np
import torch
from PIL import Image
from common import models,ROOT

@dataclass
class Observation:
    depth: np.ndarray
    semantic_probability: np.ndarray
    depth_probability: np.ndarray
    room_probability: np.ndarray
    valid_weights: np.ndarray
    latency_ms: float=0.
    peak_allocated_mib: float=0.
    peak_reserved_mib: float=0.
    def labels(self,policy='argmax',seed=42):
        if policy=='argmax':return self.semantic_probability.argmax(-1)
        if policy=='multinomial':
            gen=torch.Generator().manual_seed(seed)
            return torch.multinomial(torch.tensor(self.semantic_probability),1,generator=gen).numpy().ravel()
        raise ValueError(policy)

class ObservationModel:
    def __init__(self,device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Deliberately disallow silent contamination of a baseline by smoke weights.
        import os
        if os.environ.get('FYP_FINETUNED'):raise ValueError('Unset FYP_FINETUNED for audited official checkpoint runs')
        self.depth,self.semantic=models();self.depth.to(self.device).eval();self.semantic.to(self.device).eval()
    def predict(self,rgb_path,valid_mask=None):
        image=np.asarray(Image.open(rgb_path).convert('RGB'),dtype=np.float32)/255.
        if image.shape!=(360,640,3):raise ValueError('Require calibrated, gravity-aligned 640x360 RGB and 80deg HFOV')
        valid=np.ones((360,640),np.float32) if valid_mask is None else np.asarray(valid_mask,dtype=np.float32)
        if valid.shape!=(360,640):raise ValueError('Invalid mask shape')
        if not np.isfinite(valid).all() or valid.min()<0 or valid.max()>1:raise ValueError('Mask must be finite in [0,1]')
        if valid.mean()<.01:raise ValueError('Insufficient image support: abstain before attention softmax')
        # Image mask is 2D support, not GT dynamic-mask input in default online use.
        x=torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to(self.device);mask=torch.tensor(valid,device=self.device).unsqueeze(0)
        if self.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():d,_,dp=self.depth(x,mask);s,r,_=self.semantic(x,mask)
        if self.device=='cuda':torch.cuda.synchronize()
        if not torch.isfinite(d).all() or not torch.isfinite(s).all():raise ValueError('Non-finite network output')
        weights=torch.nn.functional.adaptive_avg_pool2d(mask.unsqueeze(1),(1,40)).cpu().numpy().ravel()
        return Observation(d[0].cpu().numpy(),s.softmax(-1)[0].cpu().numpy(),dp[0].cpu().numpy(),r.softmax(-1)[0].cpu().numpy(),weights,1000*(time.perf_counter()-start),torch.cuda.max_memory_allocated()/2**20 if self.device=='cuda' else 0,torch.cuda.max_memory_reserved()/2**20 if self.device=='cuda' else 0)
