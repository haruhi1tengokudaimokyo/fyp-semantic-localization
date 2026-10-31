import json, pathlib, sys

p = pathlib.Path(__file__).resolve().parents[2]
specs = json.loads((p / "experiment_automation/manifests/tasks.json").read_text())
ids = {s["task_id"] for s in specs}
assert len(ids) == len(specs)
for s in specs:
    assert set(s["depends_on"]) <= ids
    if s.get("model_dataset_id"):
        assert s["model_dataset_id"] == s["dataset_id"]
    if not s.get("blocked_reason") and not s.get("reuse_existing"):
        assert pathlib.Path(s["command"][0]).is_file(), s["task_id"]
        assert pathlib.Path(s["command"][1]).is_file(), s["task_id"]
        assert s["validation"], s["task_id"]
visited = set()
active = set()


def dfs(tid):
    assert tid not in active, "DAG cycle"
    if tid in visited:
        return
    active.add(tid)
    for dep in next(s for s in specs if s["task_id"] == tid)["depends_on"]:
        dfs(dep)
    active.remove(tid)
    visited.add(tid)


for tid in ids:
    dfs(tid)
print(
    "DAG dependency, interpreter, command, validation and model-scene guards passed:",
    len(specs),
    "tasks",
)
