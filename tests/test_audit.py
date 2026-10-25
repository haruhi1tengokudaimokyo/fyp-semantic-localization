"""Unit fixtures are not evaluation datasets. Integrations use the real public sample."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read_sample,models
from localization.geometry import *
from localization.floorplan.rays import cast_many
from localization.floorplan.representation import Floorplan
from localization.matching.engine import Matcher
from localization.datasets.zind_audit import audit_annotations
from localization.models.observation import ObservationModel
from utils.raycast_utils import ray_cast,get_color_name
BASE=Path(__file__).resolve().parents[1]
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.paths=json.loads((ROOT/'data/zind_public_sample/manifest.json').read_text());cls.meta,_=read_sample(cls.paths[0]);cls.floor=Floorplan(cls.meta['map'],cls.meta['free_mask']);cls.profile=json.loads((BASE/'configs/integration/zind.json').read_text());cls.matcher=Matcher(cls.floor.rays,cls.floor.image,cls.profile)
 def test_world_image_round_trip_and_scale(self):
  p=np.array([[3.,4.],[-1.,7.]]);origin=[-2.,10.];np.testing.assert_allclose(image_to_world(world_to_image(p,origin,.01),origin,.01),p)
  with self.assertRaises(ValueError):Floorplan(self.meta['map'],self.meta['free_mask'],.05)
 def test_axial_radial_roundtrip(self):
  radial=np.linspace(.1,15,40);np.testing.assert_allclose(axial_to_radial(radial_to_axial(radial)),radial);self.assertGreater(axial_to_radial(np.ones(40))[0],1.2)
 def test_independent_annotation_projection_and_crop(self):
  details,_,_=audit_annotations(self.paths);self.assertEqual(len(details),32);self.assertLess(max(x['independent_bearing_error_rad'] for x in details),1e-12);self.assertTrue(all(x['crop_max_uint8_diff']==0 for x in details));self.assertTrue(all(.5<x['camera_height_m']<3 for x in details))
 def test_semantic_palette_and_door_through(self):
  self.assertEqual([get_color_name(*x) for x in [[0,0,0],[0,0,1],[1,0,0]]],['black','blue','red'])
  image=np.ones((200,200,3));image[:,150]=0;image[:,100]=[1,0,0];d,s,h=ray_cast(image,[20,100],0);self.assertEqual(s,2);self.assertEqual(d,130)
  dr,sr,hr=cast_many(image,[[20,100]],[0]);self.assertEqual(sr[0],s);self.assertAlmostEqual(dr[0],d*.01);np.testing.assert_array_equal(hr[0],h)
 def test_fast_backend_matches_upstream_real_map(self):
  rng=np.random.default_rng(17);coords=np.argwhere(self.floor.rays['valid']);chosen=coords[rng.choice(len(coords),200,replace=False)];positions=chosen[:,::-1]*10;angles=rng.uniform(-np.pi,np.pi,len(positions));d,s,h=cast_many(self.floor.image,positions,angles)
  for i,(p,a) in enumerate(zip(positions,angles)):
   expected=ray_cast(self.floor.image,p,a);self.assertAlmostEqual(d[i],expected[0]*.01,places=12);self.assertEqual(s[i],expected[1]);np.testing.assert_array_equal(h[i],expected[2])
 def test_full_coarse_fine_gt_first_real_query(self):
  d=np.array(self.meta['depth_axial_m']);s=np.array(self.meta['semantic']);pd,ps,v=self.matcher.volumes(d,s);result,_=self.matcher.run_from_volumes(d,s,pd,ps,v);xy,yaw=pose_error(result['estimated_pose'],self.meta['pose']);self.assertLess(xy,.2);self.assertLess(yaw,6);self.assertIsNone(result['success_probability']);self.assertIsNone(result['accepted_pose'])
 def test_candidates_are_distinct_in_se2(self):
  score=np.zeros_like(self.floor.rays['depth']);valid=self.floor.rays['valid'];y,x=np.argwhere(valid)[100];score[y,x,0]=1;score[y,x,18]=.95
  if x+1<score.shape[1] and valid[y,x+1]:score[y,x+1,0]=.98
  picks=self.matcher.candidates(score,valid);self.assertEqual(len(picks),2);self.assertAlmostEqual(abs(wrap_angle(picks[0]['pose'][2]-picks[1]['pose'][2])),np.pi)
 def test_all_masked_matching_rejects(self):
  with self.assertRaises(ValueError):self.matcher.volumes(np.ones(40),np.zeros(40,int),np.zeros(40))
 def test_upstream_fine_backend_reuses_algorithm(self):
  from evaluation.candidate_refiner import compute_rays_from_candidate
  pose=np.array(self.meta['pose']);pose[:2]=np.round(pose[:2]*10)/10;pose[2]=np.pi/2
  expected=compute_rays_from_candidate(None,self.floor.image,{'x':pose[0]/.1,'y':pose[1]/.1,'orientation_radians':pose[2]},{'0':0.,'5':np.deg2rad(5),'-5':np.deg2rad(-5)});actual=self.matcher.candidate_rays(pose)
  for k in expected:np.testing.assert_allclose(actual[k][0],expected[k][0],atol=1e-12);np.testing.assert_array_equal(actual[k][1],expected[k][1])
 def test_semantic_dispatch_is_not_class_id_depth_l1(self):
  from utils.localization_utils import localize
  tensor=torch.zeros((1,2,36));tensor[:,1,:]=2;rays=torch.full((9,),2)
  semantic=localize(tensor,rays,return_np=False,localize_type='semantic')[0]
  legacy=localize(tensor,rays,return_np=False,localize_type='depth')[0]
  self.assertAlmostEqual(float(semantic[0,0,0]),float(np.exp(-9/40)),places=6)
  self.assertNotAlmostEqual(float(semantic[0,0,0]),float(legacy[0,0,0]),places=5)
 def test_previous_training_changed_heads_not_backbone(self):
  ck=torch.load(ROOT/'runs/smoke/heads_finetuned.pt',map_location='cpu',weights_only=True);base=torch.load(ROOT/'models/zind/final_depth.ckpt',map_location='cpu',weights_only=True)['state_dict'];base={k[len('encoder.'):]:v for k,v in base.items() if k.startswith('encoder.')}
  for k in base:
   if k.startswith('depth_feature.resnet.'):self.assertTrue(torch.equal(base[k],ck['depth'][k]),k)
  self.assertTrue(any(not torch.equal(base[k],ck['depth'][k]) for k in base if k.startswith('depth_feature.q_proj')));model,_=models();model.load_state_dict(ck['depth'],strict=True)
class NetworkInterface(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.model=ObservationModel();cls.path=ROOT/'data/zind_public_sample/samples/000_000/rgb.png'
 def test_no_gt_api_matches_official_weights(self):
  obs=self.model.predict(self.path);cache=json.loads((ROOT/'runs/audit_v2/perception_v1.json').read_text())[0];np.testing.assert_allclose(obs.depth,cache['pred_depth'],atol=2e-5);np.testing.assert_allclose(obs.semantic_probability,cache['semantic_prob'],atol=2e-5);self.assertEqual(obs.semantic_probability.shape,(40,4));self.assertTrue(np.isfinite(obs.depth).all())
 def test_full_occlusion_rejected(self):
  with self.assertRaises(ValueError):self.model.predict(self.path,np.zeros((360,640)))
 def test_partial_support_finite_and_not_silently_ignored(self):
  mask=np.ones((360,640));mask[:,240:400]=0;obs=self.model.predict(self.path,mask);self.assertTrue(np.isfinite(obs.depth).all());self.assertTrue((obs.valid_weights[15:25]==0).all());meta,_=read_sample(self.path.parent);f=Floorplan(meta['map'],meta['free_mask']);m=Matcher(f.rays,f.image,json.loads((BASE/'configs/integration/zind.json').read_text()))
  with self.assertRaises(ValueError):m.volumes(obs.depth,obs.labels(),obs.valid_weights)
 def test_online_full_pipeline_cannot_read_gt_loader(self):
  from unittest.mock import patch
  from localize import locate
  meta=json.loads((self.path.parent/'sample.json').read_text());profile=json.loads((BASE/'configs/integration/zind.json').read_text())
  with patch('common.read_sample',side_effect=AssertionError('GT loader must never run online')):
   result,_,_=locate(self.path,meta['map'],meta['free_mask'],profile,self.model)
  self.assertFalse(result['ground_truth_used']);self.assertIsNone(result['success_probability'])
 def test_formal_training_is_gated_without_baseline(self):
  import tempfile,subprocess
  with tempfile.TemporaryDirectory() as out:
   run=subprocess.run([sys.executable,str(BASE/'src/train.py'),'--manifest',str(ROOT/'data/zind_public_sample/manifest.json'),'--out',out],capture_output=True,text=True)
   self.assertNotEqual(run.returncode,0);self.assertIn('Formal training is gated',run.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
