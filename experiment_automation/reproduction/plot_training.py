"""Scientific curves from actual full-8M ACE worker log snapshots; no generated data."""

import csv, hashlib, io, json, re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[2] / "outputs/FYP_Automation"


def main():
    rows = []
    sources = []
    for log in sorted((ROOT / "logs").glob("R.ACE.7scenes.*.train8M.attempt*.log")):
        data = log.read_bytes()
        text = data.decode(errors="replace")
        scene = log.name.split(".")[3]
        records = re.findall(
            r"Iteration:\s+(\d+) / Epoch (\d+)\|016, Loss: ([\d.eE+-]+), Valid: ([\d.]+)%, Time: ([\d.]+)s",
            text,
        )
        for i, e, loss, valid, elapsed in records:
            rows.append(
                dict(
                    scene=scene,
                    iteration=int(i),
                    epoch=int(e),
                    loss=float(loss),
                    valid_percent=float(valid),
                    elapsed_seconds=float(elapsed),
                    log=log.name,
                )
            )
        sources.append(
            dict(
                log=str(log),
                snapshot_sha256=hashlib.sha256(data).hexdigest(),
                points=len(records),
                last_iteration=int(records[-1][0]) if records else None,
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for scene in sorted({r["scene"] for r in rows}):
            rr = [r for r in rows if r["scene"] == scene]
            axes[0].plot(
                [r["iteration"] for r in rr],
                [r["loss"] for r in rr],
                label=scene,
                lw=1.3,
            )
            axes[1].plot(
                [r["iteration"] for r in rr],
                [r["valid_percent"] for r in rr],
                label=scene,
                lw=1.3,
            )
        axes[0].set(yscale="log", ylabel="Training loss (log scale)")
        axes[1].set(ylabel="Valid reprojection pixels (%)", ylim=(0, 100))
        for ax in axes:
            ax.set_xlabel("Logged optimizer iteration")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle("ACE: 8M features, batch5120, 16epochs - actual training logs")
        path = OUT / "ACE_8M_training_progress.png"
        temp = path.with_suffix(".tmp.png")
        fig.savefig(temp, dpi=170)
        temp.replace(path)
        plt.close(fig)
        stream = io.StringIO()
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        (OUT / "ACE_8M_training_progress.csv").write_text(stream.getvalue())
    (OUT / "ACE_8M_training_curve_sources.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                scope="Training progress only, not test localization accuracy. Last log point is not an invented endpoint.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
