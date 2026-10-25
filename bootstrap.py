"""Fetch pinned upstream code and apply the documented compatibility patch.
Does not download licensed Structured3D/ZInD datasets or install dependencies.
"""
import os,subprocess
from pathlib import Path
ROOT=Path(os.environ.get('FYP_ROOT','/data/fyp-semrayloc'))
BASE=Path(__file__).resolve().parent
UPSTREAM={'SemRayLoc':('https://github.com/TAU-VAILab/SemRayLoc.git','072b7a3d6056c44ad95d08b269bdd296fc3eadab'),'zind':('https://github.com/zillow/zind.git','06cfdbf295fa21500a857efe81f80f2243b8fb40')}
def run(*a,**kw): subprocess.run(a,check=True,**kw)
for name,(url,sha) in UPSTREAM.items():
    dst=ROOT/'repos'/name
    if not dst.exists(): run('git','clone',url,str(dst))
    head=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
    if head!=sha:
        dirty=subprocess.check_output(['git','-C',str(dst),'status','--porcelain'],text=True).strip()
        if dirty: raise RuntimeError(f'{dst} contains changes; refusing checkout')
        run('git','-C',str(dst),'checkout','--detach',sha)
repo=ROOT/'repos/SemRayLoc';patch=str(BASE/'official_fixes.patch')
if subprocess.run(['git','-C',str(repo),'apply','--reverse','--check',patch],capture_output=True).returncode==0:
    print('Compatibility patch already applied')
else:
    run('git','-C',str(repo),'apply','--check',patch)
    run('git','-C',str(repo),'apply',patch)
print('Pinned source ready. Install requirements and fetch author checkpoints as described in README.')
