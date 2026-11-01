"""Accept core localization only from independently checked complete trajectories."""
import argparse,json
from pathlib import Path
from blocked_data import atomic
def main(output,inputs):
    values=[]
    for p in inputs:
        v=json.loads(Path(p).read_text());assert v['status']=='accepted' and v['localization_executed'] and v['independently_recomputed'] and v['frames']==600
        values.append(dict(path=p,metrics=v))
    assert len(values)>=2
    atomic(output,dict(status='accepted',localization_executed=True,independent_metrics=True,results=values,not_original_paper_dataset=True,scope='Original method evaluated on two ReplicaCAD layouts of the shared FRL apartment; not original-paper dataset reproduction or independent physical buildings'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('inputs',nargs='+');a=p.parse_args();main(a.output,a.inputs)
