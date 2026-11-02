"""Persist the user's narrowed experiment scope without erasing historical states."""
import json

def apply_scope(tasks,path):
    if not path.exists():return tasks
    policy=json.loads(path.read_text());by={t['task_id']:t for t in tasks}
    keep={t['task_id'] for t in tasks if t.get('paper_id') in policy['active_paper_ids'] or t.get('experiment_id') in policy['active_legacy_experiment_ids']}
    pending=list(keep)
    while pending:
        for dep in by[pending.pop()].get('depends_on',[]):
            if dep in by and dep not in keep:keep.add(dep);pending.append(dep)
    for task in tasks:
        if task['task_id'] not in keep:task['blocked_reason']='RETIRED_BY_USER_SCOPE: '+policy['reason']
    return tasks
