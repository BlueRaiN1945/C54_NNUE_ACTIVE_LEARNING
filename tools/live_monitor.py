#!/usr/bin/env python3

import argparse
import re
import subprocess
import time
from pathlib import Path


TRAIN_PROGRESS_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\(Train\):\s+"
    r"(?P<pct>\d+)%\|\s*"
    r"(?P<cur>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<,\]]+)<(?P<eta>[^,\]]+),\s*"
    r"(?P<rate>[0-9.]+)it/s,\s*"
    r"train_loss=(?P<loss>[-+0-9.eE]+)"
)

TRAIN_EPOCH_RE = re.compile(
    r"train_loss_epoch=([-+0-9.eE]+)"
)

VAL_EPOCH_RE = re.compile(
    r"val_loss_epoch=([-+0-9.eE]+)"
)

FIXED_VAL_RE = re.compile(
    r"Mean validation loss:\s*([-+0-9.eE]+)"
)

BATCH_RE = re.compile(
    r"batch_size\(global\)=(\d+)"
)

LR_RE = re.compile(
    r"optimizer with lr:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)


def read_tail(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""

    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def last_float(regex, text):
    matches = regex.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def find_fixed_val(run: Path):
    files = sorted(
        run.glob("eval*.txt"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )

    for path in reversed(files):
        text = read_tail(path)
        m = FIXED_VAL_RE.search(text)
        if m:
            return float(m.group(1)), path.name

    return None, None


def parse_run(run: Path):
    train_log = run / "train.log"
    text = read_tail(train_log)

    progress_matches = list(TRAIN_PROGRESS_RE.finditer(text))
    progress = None

    if progress_matches:
        d = progress_matches[-1].groupdict()
        progress = {
            "epoch": int(d["epoch"]),
            "pct": int(d["pct"]),
            "cur": int(d["cur"]),
            "total": int(d["total"]),
            "elapsed": d["elapsed"].strip(),
            "eta": d["eta"].strip(),
            "rate": float(d["rate"]),
            "loss": float(d["loss"]),
        }

    train_epoch_loss = last_float(TRAIN_EPOCH_RE, text)
    val_epoch_loss = last_float(VAL_EPOCH_RE, text)

    batch = None
    m = BATCH_RE.search(text)
    if m:
        batch = int(m.group(1))

    lr = None
    m = LR_RE.search(text)
    if m:
        lr = float(m.group(1).rstrip(".,;:"))

    fixed_val, eval_file = find_fixed_val(run)

    candidate_pts = list(run.glob("candidate*.pt"))
    finished = (run / "training_finished").exists()

    eval_files = list(run.glob("eval*.txt"))
    has_eval_file = bool(eval_files)

    if fixed_val is not None:
        status = "COMPLETE"
    elif has_eval_file or candidate_pts:
        status = "EVALUATING"
    elif finished:
        status = "SERIALIZING"
    elif progress is not None:
        if progress["cur"] >= progress["total"]:
            status = "FINISHING"
        else:
            status = "TRAINING"
    elif train_log.exists():
        status = "STARTING"
    else:
        status = "PENDING"

    return {
        "name": run.name,
        "path": run,
        "status": status,
        "progress": progress,
        "train_epoch_loss": train_epoch_loss,
        "val_epoch_loss": val_epoch_loss,
        "fixed_val": fixed_val,
        "eval_file": eval_file,
        "batch": batch,
        "lr": lr,
    }


def discover_runs(root: Path):
    if (
        (root / "train.log").exists()
        or list(root.glob("eval*.txt"))
    ):
        return [root]

    runs = []

    if not root.exists():
        return runs

    for p in root.iterdir():
        if not p.is_dir():
            continue

        if (
            (p / "train.log").exists()
            or (p / "training_finished").exists()
            or list(p.glob("eval*.txt"))
            or list(p.glob("candidate*.pt"))
        ):
            runs.append(p)

    return sorted(runs, key=lambda p: p.name)


def gpu_info():
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,"
        "memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )

        if r.returncode != 0 or not r.stdout.strip():
            return None

        fields = [
            x.strip()
            for x in r.stdout.strip().splitlines()[0].split(",")
        ]

        if len(fields) < 6:
            return None

        return {
            "name": fields[0],
            "util": fields[1],
            "mem_used": fields[2],
            "mem_total": fields[3],
            "temp": fields[4],
            "power": fields[5],
        }

    except Exception:
        return None


def progress_bar(progress, width=18):
    if progress is None:
        return "[" + ("-" * width) + "]"

    total = max(1, progress["total"])
    cur = max(0, min(progress["cur"], total))

    filled = int(round(width * cur / total))
    return (
        "["
        + ("#" * filled)
        + ("-" * (width - filled))
        + "]"
    )


def fmt_float(value, digits=8):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_delta(value, baseline):
    if value is None or baseline is None:
        return "-"

    pct = 100.0 * (value - baseline) / baseline
    return f"{pct:+.4f}%"


def clear_screen():
    print("\033[2J\033[H", end="")


def render(root, args):
    runs = [parse_run(p) for p in discover_runs(root)]
    gpu = None if args.no_gpu else gpu_info()

    clear_screen()

    print("=" * 118)
    print("C54 NNUE LIVE MONITOR")
    print("=" * 118)

    print(f"Experiment : {root}")

    if gpu:
        print(
            "GPU        : "
            f"{gpu['name']} | "
            f"util {gpu['util']}% | "
            f"VRAM {gpu['mem_used']}/{gpu['mem_total']} MiB | "
            f"{gpu['temp']} C | "
            f"{gpu['power']} W"
        )
    else:
        print("GPU        : unavailable")

    print(
        f"Official   : {args.official_loss:.10f} | "
        f"{args.reference_label}: {args.reference_loss:.10f}"
    )

    print()
    print(
        f"{'Run':<18}"
        f"{'Status':<13}"
        f"{'Progress':<31}"
        f"{'Train':>11}"
        f"{'Int.Val':>11}"
        f"{'Fixed VAL':>13}"
        f"{'vs Off':>11}"
        f"{'vs Ref':>11}"
        f"{'ETA':>9}"
    )
    print("-" * 118)

    completed = []

    for r in runs:
        p = r["progress"]

        if p:
            prog = (
                f"{progress_bar(p)} "
                f"{p['cur']}/{p['total']}"
            )
            current_train = p["loss"]
            eta = p["eta"]
        else:
            prog = progress_bar(None)
            current_train = r["train_epoch_loss"]
            eta = "-"

        fixed = r["fixed_val"]

        if fixed is not None:
            completed.append(r)

        print(
            f"{r['name']:<18}"
            f"{r['status']:<13}"
            f"{prog:<31}"
            f"{fmt_float(current_train, 5):>11}"
            f"{fmt_float(r['val_epoch_loss'], 5):>11}"
            f"{fmt_float(fixed, 10):>13}"
            f"{fmt_delta(fixed, args.official_loss):>11}"
            f"{fmt_delta(fixed, args.reference_loss):>11}"
            f"{eta:>9}"
        )

    if not runs:
        print("No runs discovered yet.")

    print()

    if completed:
        best = min(completed, key=lambda r: r["fixed_val"])

        print(
            "Best completed run: "
            f"{best['name']} = {best['fixed_val']:.10f} "
            f"({fmt_delta(best['fixed_val'], args.official_loss)} "
            "vs official)"
        )

        if best["batch"] is not None:
            print(f"Batch size        : {best['batch']}")

        if best["lr"] is not None:
            print(f"Learning rate     : {best['lr']:.8g}")

        if best["eval_file"]:
            print(f"Evaluation file   : {best['eval_file']}")

    active = [
        r for r in runs
        if r["status"] not in ("COMPLETE", "PENDING")
    ]

    if active:
        r = active[-1]
        print()
        print(f"Active run        : {r['name']}")

        if r["batch"] is not None:
            print(f"Batch size        : {r['batch']}")

        if r["lr"] is not None:
            print(f"Learning rate     : {r['lr']:.8g}")

        if r["progress"]:
            p = r["progress"]
            print(
                f"Epoch progress    : "
                f"{p['cur']}/{p['total']} "
                f"({p['pct']}%)"
            )
            print(f"Elapsed           : {p['elapsed']}")
            print(f"ETA               : {p['eta']}")
            print(f"Iterations/sec    : {p['rate']:.2f}")

    print()
    print(
        "Ctrl+C stops only this monitor; "
        "the training process continues."
    )
    print("=" * 118)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only live monitor for C54 NNUE experiments."
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Experiment root or single run directory.",
    )

    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0).",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit.",
    )

    parser.add_argument(
        "--official-loss",
        type=float,
        default=0.0067736587,
    )

    parser.add_argument(
        "--reference-loss",
        type=float,
        default=0.0067599019,
    )

    parser.add_argument(
        "--reference-label",
        default="Old best",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
    )

    args = parser.parse_args()

    root = args.experiment.resolve()

    try:
        while True:
            render(root, args)

            if args.once:
                break

            time.sleep(max(0.2, args.refresh))

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
