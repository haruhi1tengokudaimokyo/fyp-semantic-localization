import argparse, csv, json, re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(a):
    text = Path(a.log).read_text()
    pattern = r"Iteration:\s+(\d+) / Epoch (\d+)\|\d+, Loss: ([\d.]+), Valid: ([\d.]+)%, Time: ([\d.]+)s"
    rows = [
        dict(
            iteration=int(i),
            epoch=int(e),
            training_loss=float(l),
            valid_percent=float(v),
            seconds=float(s),
        )
        for i, e, l, v, s in re.findall(pattern, text)
    ]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training_curve.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["iteration"] for r in rows]
    axs[0].plot(x, [r["training_loss"] for r in rows], color="#5068a4")
    axs[0].set_ylabel("Official training loss (logged batches)")
    axs[1].plot(x, [r["valid_percent"] for r in rows], color="#208a85")
    axs[1].set_ylabel("Valid scene-coordinate predictions (%)")
    for ax in axs:
        ax.set_xlabel("Optimizer batch iteration")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Heads train sequence only | ACE 1M buffer / 16 epochs | not validation metrics"
    )
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "training_curve.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("output")
    main(p.parse_args())
