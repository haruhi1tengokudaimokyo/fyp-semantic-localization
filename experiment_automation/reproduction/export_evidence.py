"""Export small real evidence and model receipt indexes; never copy large weights/data."""

import csv, hashlib, io, json, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/data/fyp_3d")
OUT = ROOT.parents[2] / "outputs/FYP_Automation"


def atomic_bytes(p, b):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_bytes(b)
    t.replace(p)


def main():
    sources = [
        BASE / "runs/reproduction/protocol_comparison.json",
        BASE / "runs/reproduction/ACE/data_audit_7scenes.json",
        BASE / "runs/reproduction/MACVO/official_metrics_existing_demo_v2.json",
        BASE / "runs/reproduction/OMCL/lseg_batch1_probe.json",
        BASE / "runs/reproduction/SG2Loc/kernel_probe.json",
        BASE / "runs/reproduction/OMCL/kernel_probe.json",
        ROOT / "artifacts/reproduction_timer_test.json",
        ROOT / "artifacts/hypersim_ai027003_diagnostic.json",
        ROOT / "artifacts/hypersim_ai043009_affine_probe.json",
        ROOT / "artifacts/12scenes_source_endpoint_recheck.json",
    ]
    for paper in ["MACVO", "ACE", "SG2Loc", "OMCL"]:
        sources.append(BASE / "runs/reproduction" / paper / "source_audit.json")
    models = []
    failures = []
    with sqlite3.connect(
        f'file:{ROOT/"state/experiments.sqlite"}?mode=ro', uri=True
    ) as c:
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT * FROM tasks"):
            if r["status"] == "COMPLETED":
                v = json.loads(r["validation"] or "{}")
                for item in v.get("artifacts", []):
                    p = Path(item["path"])
                    if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".safetensors"):
                        models.append(
                            dict(
                                task_id=r["task_id"],
                                paper_id=r["paper_id"],
                                reproduction_level=r["reproduction_level"],
                                path=str(p),
                                receipt_sha256=item["sha256"],
                                validated_bytes=item["bytes"],
                                exists_now=p.is_file(),
                                current_bytes=p.stat().st_size if p.is_file() else None,
                                hash_scope="actual task acceptance receipt; not rehashed during report generation",
                                validation_utc=v.get("validated_utc"),
                            )
                        )
            if r["status"] in ("FAILED", "BLOCKED", "RETRYING"):
                failure = r["failure"] or ""
                try:
                    message = json.loads(failure).get("message", failure)
                except (ValueError, TypeError):
                    message = failure
                failures.append(
                    dict(
                        task_id=r["task_id"],
                        status=r["status"],
                        attempt=r["attempt"],
                        reason=message,
                        log=(
                            str(
                                ROOT
                                / "logs"
                                / f"{r['task_id']}.attempt{r['attempt']}.log"
                            )
                            if r["attempt"]
                            else None
                        ),
                    )
                )
    index = []
    for n, p in enumerate(sources):
        if not p.is_file():
            continue
        label = p.parent.name + "__" + p.name
        dest = OUT / "source_evidence" / label
        b = p.read_bytes()
        atomic_bytes(dest, b)
        index.append(
            dict(
                source=str(p),
                export=str(dest),
                sha256=hashlib.sha256(b).hexdigest(),
                bytes=len(b),
            )
        )
    atomic_bytes(
        OUT / "source_evidence_index.json",
        json.dumps(index, ensure_ascii=False, indent=2).encode(),
    )
    atomic_bytes(
        OUT / "model_weights_index.json",
        json.dumps(models, ensure_ascii=False, indent=2).encode(),
    )
    stream = io.StringIO()
    w = csv.DictWriter(
        stream, fieldnames=["task_id", "status", "attempt", "reason", "log"]
    )
    w.writeheader()
    w.writerows(failures)
    atomic_bytes(OUT / "failure_and_blocker_index.csv", stream.getvalue().encode())


if __name__ == "__main__":
    main()
