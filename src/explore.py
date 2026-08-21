"""Interactive tlog explorer — run from the project root."""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TLOG_DIR


# ── helpers ────────────────────────────────────────────────────────────────────

def pick(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:>3}. {opt}")
    while True:
        raw = input("\n> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"Enter a number between 1 and {len(options)}.")


def scan_types(path: Path) -> Counter:
    """First pass: count message types."""
    print(f"\nScanning {path.name} …", end="", flush=True)
    counts: Counter = Counter()
    log = mavutil.mavlink_connection(str(path))
    while True:
        msg = log.recv_match(blocking=False)
        if msg is None:
            break
        counts[msg.get_type()] += 1
    print(f" {sum(counts.values())} messages, {len(counts)} types.")
    return counts


def load_messages(path: Path, msg_type: str) -> pd.DataFrame:
    """Second pass: load all messages of a given type into a DataFrame."""
    print(f"Loading {msg_type} messages …", end="", flush=True)
    rows = []
    log = mavutil.mavlink_connection(str(path))
    while True:
        msg = log.recv_match(type=msg_type, blocking=False)
        if msg is None:
            break
        d = msg.to_dict()
        d["_timestamp"] = msg._timestamp
        rows.append(d)
    df = pd.DataFrame(rows)
    if not df.empty and "_timestamp" in df.columns:
        df["_timestamp"] = pd.to_datetime(df["_timestamp"], unit="s")
        df = df.set_index("_timestamp").sort_index()
    print(f" {len(df)} rows.")
    return df


def show(df: pd.DataFrame, page_size: int = 20) -> None:
    """Page through a DataFrame."""
    total = len(df)
    start = 0
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)

    while True:
        chunk = df.iloc[start : start + page_size]
        print(f"\n── rows {start + 1}–{min(start + page_size, total)} of {total} ──")
        print(chunk.to_string())
        print("\n[n] next  [p] prev  [c] columns  [s] stats  [q] back")
        cmd = input("> ").strip().lower()
        if cmd == "n":
            if start + page_size < total:
                start += page_size
            else:
                print("Already at end.")
        elif cmd == "p":
            start = max(0, start - page_size)
        elif cmd == "c":
            print("\nColumns:", ", ".join(df.columns.tolist()))
        elif cmd == "s":
            print(df.describe(include="all").to_string())
        elif cmd == "q":
            break


# ── main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    tlog_files = sorted(TLOG_DIR.glob("*.tlog"))
    if not tlog_files:
        print(f"No .tlog files found in {TLOG_DIR}")
        return

    while True:
        file_names = [f.name for f in tlog_files]
        chosen_name = pick("Select a tlog file:", file_names)
        chosen_path = TLOG_DIR / chosen_name

        counts = scan_types(chosen_path)
        type_labels = [
            f"{t:<30} ({n} msgs)"
            for t, n in sorted(counts.items(), key=lambda x: -x[1])
        ]
        raw_types = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])]

        while True:
            label = pick("Select a message type:", type_labels)
            msg_type = raw_types[type_labels.index(label)]

            df = load_messages(chosen_path, msg_type)
            if df.empty:
                print("No data found for that type.")
            else:
                show(df)

            again = input("\n[t] pick another type  [f] pick another file  [q] quit\n> ").strip().lower()
            if again == "q":
                return
            elif again == "f":
                break


if __name__ == "__main__":
    main()
