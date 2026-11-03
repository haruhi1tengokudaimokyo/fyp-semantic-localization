"""Preserve scalar float supervision through torchvision/PIL conversion.

RGB transformations, original sampling, network, losses and optimization stay
unchanged. Explicit PIL F mode prevents uint8 wrapping of class IDs, metre
depths, and negative normal components in the upstream per-channel transforms.
"""
import importlib
from torchvision.transforms import ToPILImage

def install():
    data = importlib.import_module('spvloc.data.dataset')
    model = importlib.import_module('spvloc.model.spvloc')
    original = data.Structured3DPlans_Perspective
    class FloatTargetDataset(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for name in ['layout_transform', 'semantic_transform', 'normal_layout_transform']:
                transform = getattr(self, name)
                assert isinstance(transform.transforms[0], ToPILImage)
                transform.transforms[0] = ToPILImage(mode='F')
    data.Structured3DPlans_Perspective = FloatTargetDataset
    model.Structured3DPlans_Perspective = FloatTargetDataset
    return FloatTargetDataset
