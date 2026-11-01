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
for s in specs:
    cmd = s.get("command", [])
    if s["task_id"].startswith("R.ACE.") and s["task_id"].endswith(".train8M"):
        dataset = pathlib.Path(cmd[2])
        output = pathlib.Path(cmd[3])
        if s['task_id']=='R.ACE.7scenes.chess.train8M':
            assert s['dataset_id']==s['task_id'] and dataset.name=='7scenes_chess'
        else:
            assert s["dataset_id"] == dataset.name, s["task_id"]
        assert all(
            pathlib.Path(v["path"]).is_relative_to(output) for v in s["validation"]
        ), s["task_id"]
    if len(cmd) > 1 and pathlib.Path(cmd[1]).name == "official_ace_eval.py":
        dataset = pathlib.Path(cmd[3])
        model = pathlib.Path(cmd[4])
        output = pathlib.Path(cmd[5])
        assert s["dataset_id"] == dataset.name, s["task_id"]
        assert all(
            pathlib.Path(v["path"]).is_relative_to(output) for v in s["validation"]
        ), s["task_id"]
        scene_token = dataset.name.removeprefix("7scenes_")
        assert scene_token in str(model), s["task_id"]

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
