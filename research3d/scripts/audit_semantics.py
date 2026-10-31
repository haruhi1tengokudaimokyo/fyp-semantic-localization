"""Measured semantic coverage. ID 0 is unknown, never implicitly a wall label."""
import argparse,json
from pathlib import Path

def main(a):
 root=Path(a.root);lex=json.loads((root/'semantics/hssd-hab_semantic_lexicon.json').read_text());classes={x['id']:x['name'] for x in lex['classes']};configs={p.name.removesuffix('.object_config.json'):json.loads(p.read_text()) for p in root.glob('objects/**/*.object_config.json')};out=[]
 for run in sorted(Path(a.runs).glob('hssd_*')):
  if not(run/'acceptance.json').exists():continue
  d=json.loads((run/'acceptance.json').read_text());scene=json.loads((root/'scenes'/(d['scene']+'.scene_instance.json')).read_text());counts=d['semantic_pixel_counts'];total=sum(counts.values());actual={classes.get(int(k),'unmapped_id_'+k):v for k,v in counts.items()}
  absent=[];objs=[]
  for obj in scene.get('object_instances',[]):
   name=obj['template_name'];cfg=configs.get(name)
   if cfg is None:absent.append(name);continue
   objs.append({'template':name,'semantic_id':cfg.get('semantic_id',0),'category':classes.get(cfg.get('semantic_id',0),'unknown')})
  out.append(dict(scene=d['scene'],views=len(d['views']),pixel_counts=actual,unknown_pixel_fraction=counts.get('0',0)/total,missing_object_configs=absent,object_instances=objs,requested_landmark_pixels={k:actual.get(k,0) for k in ['wall','door','window','column']},interpretation='native semantic sensor gives configured class IDs for these rigid objects; IDs do not distinguish repeated instances. No architectural labels inferred for stage pixels.'))
 report=dict(scenes=out,training_decision='Do not start wall/door/window segmentation training from these native outputs: verify missing architectural labels first. Object classes with verified pixels can support a revised taxonomy; 3 scenes alone are insufficient for a robust multi-building benchmark.',split_warning='Distinct scene IDs verified. Physical building identity / duplicated layouts require further metadata audit before claiming independent buildings.')
 Path(a.output).write_text(json.dumps(report,indent=2));print([(x['scene'],x['unknown_pixel_fraction'],x['requested_landmark_pixels']) for x in out])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('runs');p.add_argument('output');main(p.parse_args())
