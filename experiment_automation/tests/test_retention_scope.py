import json,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduction'))
from retention_scope import apply_scope

def test_retirement_preserves_active_specs_and_dependency_closure(tmp_path):
    p=tmp_path/'scope.json';p.write_text(json.dumps(dict(active_paper_ids=['ACE','MACVO'],active_legacy_experiment_ids=['A','B'],reason='user scope')))
    tasks=[dict(task_id='ace',paper_id='ACE',command=['train'],depends_on=['shared']),dict(task_id='shared',command=['prepare']),dict(task_id='other',paper_id='SG2Loc',command=['localize'])]
    before=copy.deepcopy(tasks);result=apply_scope(tasks,p)
    assert result[0]==before[0] and result[1]==before[1]
    assert result[2]['command']==before[2]['command'] and result[2]['blocked_reason'].startswith('RETIRED_BY_USER_SCOPE')
