"""Check actual RGB cubemaps against independent native perspective captures.

Uses the upstream pano projection with the recorded CV camera rotation, never a
fitted yaw/translation. Thresholds fixed before the enlarged experiment.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '/data/fyp_3d/sources/spvloc')
from spvloc.utils.projection import pano2persp_from_rotation

def qualify(scene):
    receipt = json.loads((scene / 'capture_receipt.json').read_text())
    rows = []
    # canonical XYZ=(Unity x,z,y); equirectangular camera axes x,-Z,Y.
    basis = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
    for rec in receipt['records'][::8]:
        query = scene / rec['path']
        room = query.parents[2]
        pano = np.asarray(Image.open(room / 'panorama/full/rgb_rawlight.png').convert('RGB'))
        image = np.asarray(Image.open(query / 'rgb_rawlight.png').convert('RGB'))
        transform = np.loadtxt(query / 'camera_to_world_mm.txt')
        origin = np.loadtxt(room / 'panorama/camera_xyz.txt')
        assert np.linalg.norm(origin-transform[:,3]) < .01, 'Noncentral panorama'
        projected = pano2persp_from_rotation(pano, np.eye(3), basis @ transform[:,:3], 256, 256)
        # Exclude 8-pixel image edge; fixed tolerances include cubemap resampling.
        err = np.abs(projected[8:-8,8:-8].astype(float)-image[8:-8,8:-8].astype(float))
        rows.append(dict(query=rec['query'], mean_absolute_rgb_error=float(err.mean())))
    assert len(rows) >= 10
    passed = float(np.median([r['mean_absolute_rgb_error'] for r in rows])) <= 25
    result = dict(status='PANORAMA_READY' if passed else 'PANORAMA_REJECTED',
                  probes=rows, threshold_median_rgb_mae_0_255=25,
                  upstream_projection=True, fitted_transform=False)
    path = scene / 'panorama_qualification.json'
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)); tmp.replace(path)
    return result

if __name__ == '__main__':
    result = qualify(Path(sys.argv[1]))
    print(result['status'], np.median([r['mean_absolute_rgb_error'] for r in result['probes']]))
    sys.exit(0 if result['status'] == 'PANORAMA_READY' else 65)
