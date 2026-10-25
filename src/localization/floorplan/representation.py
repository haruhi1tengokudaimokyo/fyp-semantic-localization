from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from matching import atlas

class Floorplan:
    """Known, immutable map. This class has no query ground-truth fields."""
    def __init__(self,map_path,free_mask_path,meters_per_pixel=.01,step=.1):
        self.path=Path(map_path);self.free_path=Path(free_mask_path)
        self.image=np.asarray(Image.open(map_path).convert('RGB'),dtype=np.float32)/255
        self.rays=atlas({'map':str(map_path),'free_mask':str(free_mask_path),'map_m_per_pixel':meters_per_pixel},step)
        self.hash=hashlib.sha256(self.path.read_bytes()).hexdigest()
