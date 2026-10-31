#!/usr/bin/env bash
# Fresh isolated environment only. Does not touch ROS or existing environments.
set -euo pipefail
FYP3D_ROOT="${FYP3D_ROOT:-/data/fyp_3d}"
FYP3D_PYTHON="${FYP3D_PYTHON:-python3.10}"
FYP3D_SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$FYP3D_ROOT/sources" "$FYP3D_ROOT/envs" "$FYP3D_ROOT/runs"
df -h "$FYP3D_ROOT"
if [ ! -d "$FYP3D_ROOT/sources/ace/.git" ]; then
  git clone https://github.com/nianticlabs/ace "$FYP3D_ROOT/sources/ace"
fi
test "$(git -C "$FYP3D_ROOT/sources/ace" rev-parse HEAD)" = e9e90f2d02ee92c348bf411a5a60e230af6c315e
if ! git -C "$FYP3D_ROOT/sources/ace" apply --reverse --check "$FYP3D_SOURCE/docs/ace_compat.patch" 2>/dev/null; then
  git -C "$FYP3D_ROOT/sources/ace" apply "$FYP3D_SOURCE/docs/ace_compat.patch"
fi
"$FYP3D_PYTHON" -m venv "$FYP3D_ROOT/envs/ace"
"$FYP3D_ROOT/envs/ace/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$FYP3D_ROOT/envs/ace/bin/pip" install numpy==1.26.4 opencv-python-headless==4.11.0.86 scipy==1.15.3 scikit-image==0.25.2 scikit-learn==1.7.2 joblib==1.6.0 matplotlib==3.10.9 trimesh==5.2.0 pyrender==0.1.45 rtree==1.4.1 ninja requests
(
 cd "$FYP3D_ROOT/sources/ace/dsacstar"
 MAX_JOBS=1 "$FYP3D_ROOT/envs/ace/bin/python" setup.py build_ext --inplace
)
