"""
LeRobot v3 dataset format compliance test.

Validates DataCollector (controller_teleop) and DAggerCollector (deploy_bluerov)
against the four fixes required by the training team:

  Fix 1: meta/tasks.parquet + meta/episodes/chunk-000/file-000.parquet with
          full routing schema (data/file_index, dataset_from/to_index, etc.)
  Fix 2: info.json path templates use {file_index}, not {episode_index}
  Fix 3: timestamp[i] == i / fps  (grid-aligned to constant-fps video)
  Fix 4: meta/stats.json present with mean/std/min/max/quantiles per feature

Also runs lerobot.datasets.utils load functions as the final acceptance gate.

Run:  python test_dataset_format.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

SEND_HZ   = 50
CAM_W, CAM_H = 854, 480
FAKE_FRAME = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
FAKE_TELEM = {"roll": 0.1, "pitch": -0.05, "yaw": 1.2, "depth": 0.5}

STAT_KEYS = {"mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"}

# ── helpers ────────────────────────────────────────────────────────────────────

def _check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return False
    print(f"  ok  : {msg}")
    return True


def _run_episode(collector_cls, root, task, n_frames, **col_kwargs):
    col = collector_cls(task, root, **col_kwargs)
    col.start()
    for i in range(n_frames):
        x, y, z, r = i % 100, -(i % 50), 500, i % 30
        if collector_cls.__name__ == "DAggerCollector":
            col.add_frame(FAKE_FRAME, FAKE_TELEM, x, y, z, r,
                          cam_tilt=1500, gripper_closed=(i % 10 < 5),
                          is_intervention=(i % 3 == 0))
        else:
            col.add_frame(FAKE_FRAME, FAKE_TELEM, x, y, z, r,
                          cam_tilt=1500, gripper_closed=(i % 10 < 5))
    col.finish(success=True)
    return col


# ── Fix 1: tasks.parquet ──────────────────────────────────────────────────────

def check_tasks_parquet(root: Path) -> bool:
    path = root / "meta" / "tasks.parquet"
    ok = _check(path.exists(), "meta/tasks.parquet exists")
    if not ok:
        return False
    t = pq.read_table(str(path))
    ok &= _check("task_index" in t.schema.names, "tasks.parquet has task_index column")
    ok &= _check("task"       in t.schema.names, "tasks.parquet has task column")
    ok &= _check(len(t) >= 1,                    "tasks.parquet has at least one row")
    return ok


# ── Fix 1: episodes parquet ───────────────────────────────────────────────────

REQUIRED_EP_COLS = [
    "episode_index", "tasks", "length",
    "data/chunk_index", "data/file_index",
    "dataset_from_index", "dataset_to_index",
    "videos/observation.images.camera/chunk_index",
    "videos/observation.images.camera/file_index",
    "videos/observation.images.camera/from_timestamp",
    "videos/observation.images.camera/to_timestamp",
    "meta/episodes/chunk_index", "meta/episodes/file_index",
]

def check_episodes_parquet(root: Path, total_episodes: int,
                            total_frames: int, fps: float) -> bool:
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ok = _check(path.exists(), "meta/episodes/chunk-000/file-000.parquet exists")
    if not ok:
        return False

    t = pq.read_table(str(path))
    ok &= _check(len(t) == total_episodes,
                 f"episodes parquet row count={len(t)} (want {total_episodes})")

    for col in REQUIRED_EP_COLS:
        ok &= _check(col in t.schema.names, f"episodes parquet has column '{col}'")

    if "dataset_from_index" in t.schema.names and "dataset_to_index" in t.schema.names:
        froms = t["dataset_from_index"].to_pylist()
        tos   = t["dataset_to_index"].to_pylist()
        ok &= _check(froms[0] == 0, "dataset_from_index[0] == 0")
        ok &= _check(tos[-1] == total_frames,
                     f"dataset_to_index[-1]={tos[-1]} == total_frames={total_frames}")
        ok &= _check(all(tos[i] == froms[i+1] for i in range(len(tos)-1)),
                     "dataset_to[i] == dataset_from[i+1] (contiguous)")

    if "videos/observation.images.camera/to_timestamp" in t.schema.names:
        lengths = t["length"].to_pylist()
        to_ts   = t["videos/observation.images.camera/to_timestamp"].to_pylist()
        ok &= _check(
            all(abs(to_ts[i] - lengths[i] / fps) < 1e-4 for i in range(len(lengths))),
            "videos/to_timestamp == length / fps for each episode",
        )

    if "data/file_index" in t.schema.names:
        file_idxs = t["data/file_index"].to_pylist()
        ep_idxs   = t["episode_index"].to_pylist()
        ok &= _check(file_idxs == ep_idxs,
                     "data/file_index == episode_index (one file per episode)")

    return ok


# ── Fix 2: info.json path templates ──────────────────────────────────────────

def check_info_json(root: Path, total_ep: int, total_fr: int) -> bool:
    info_path = root / "meta" / "info.json"
    ok = _check(info_path.exists(), "info.json exists")
    if not ok:
        return False
    info = json.loads(info_path.read_text())
    ok &= _check(info.get("codebase_version") == "v3.0", "codebase_version=v3.0")
    ok &= _check(info.get("total_episodes") == total_ep,
                 f"total_episodes={info.get('total_episodes')} (want {total_ep})")
    ok &= _check(info.get("total_frames") == total_fr,
                 f"total_frames={info.get('total_frames')} (want {total_fr})")
    ok &= _check(isinstance(info.get("fps"), float), f"fps is float (got {info.get('fps')})")

    dp = info.get("data_path", "")
    ok &= _check("file_index" in dp and "episode_index" not in dp,
                 f"data_path uses file_index: {dp}")

    vp = info.get("video_path", "")
    ok &= _check("file_index" in vp and "episode_index" not in vp,
                 f"video_path uses file_index: {vp}")
    ok &= _check("video_key" in vp, f"video_path has video_key: {vp}")

    # template must resolve to an existing file for episode 0
    try:
        chunks_size = info.get("chunks_size", 1000)
        resolved_d = dp.format(chunk_index=0 // chunks_size, file_index=0)
        resolved_v = vp.format(video_key="observation.images.camera",
                                chunk_index=0 // chunks_size, file_index=0)
        ok &= _check((root / resolved_d).exists(),
                     f"data_path template resolves to existing file: {resolved_d}")
        ok &= _check((root / resolved_v).exists(),
                     f"video_path template resolves to existing file: {resolved_v}")
    except KeyError as e:
        ok &= _check(False, f"template format raised KeyError: {e}")

    return ok


# ── Fix 3: timestamps on fps grid ─────────────────────────────────────────────

def check_timestamp_grid(parquet_path: Path, fps: float) -> bool:
    t  = pq.read_table(str(parquet_path), columns=["timestamp", "frame_index"])
    ts = np.array(t["timestamp"].to_pylist(), dtype=np.float64)
    fi = np.array(t["frame_index"].to_pylist(), dtype=np.float64)
    expected = fi / fps
    max_err  = float(np.abs(ts - expected).max()) if len(ts) else 0.0
    return _check(max_err < 1e-4,
                  f"timestamps == frame_index/fps  (max_err={max_err:.2e})")


# ── Fix 4: stats.json ─────────────────────────────────────────────────────────

def check_stats_json(root: Path, numeric_features: list[str]) -> bool:
    path = root / "meta" / "stats.json"
    ok = _check(path.exists(), "meta/stats.json exists")
    if not ok:
        return False
    stats = json.loads(path.read_text())
    for feat in numeric_features:
        ok &= _check(feat in stats, f"stats.json has entry for '{feat}'")
        if feat in stats:
            missing = STAT_KEYS - set(stats[feat].keys())
            ok &= _check(not missing,
                         f"stats[{feat}] has all keys (missing: {missing})")
    return ok


# ── parquet schema / values ───────────────────────────────────────────────────

def check_parquet_schema(parquet_path: Path, extra_cols=()) -> bool:
    table = pq.read_table(str(parquet_path))
    schema = table.schema
    ok = True
    for col_name, expected_type in [
        ("timestamp",     pa.float32()),
        ("frame_index",   pa.int64()),
        ("episode_index", pa.int64()),
        ("index",         pa.int64()),
        ("task_index",    pa.int64()),
        ("next.done",     pa.bool_()),
        *[(c, pa.bool_()) for c in extra_cols],
    ]:
        field = schema.field(col_name) if col_name in schema.names else None
        if field is None:
            ok &= _check(False, f"column '{col_name}' present")
        else:
            ok &= _check(field.type == expected_type,
                         f"'{col_name}' type={field.type} (want {expected_type})")
    return ok


def check_parquet_values(parquet_path: Path, ep_index: int,
                         global_offset: int, n_frames: int) -> bool:
    table = pq.read_table(str(parquet_path))
    ok = True
    ok &= _check(len(table) == n_frames, f"row count={len(table)} (want {n_frames})")

    done = table["next.done"].to_pylist()
    ok &= _check(not any(done[:-1]), "next.done False for all but last frame")
    ok &= _check(done[-1],           "next.done True on last frame")

    fi = table["frame_index"].to_pylist()
    ok &= _check(fi == list(range(n_frames)), "frame_index 0..N-1")

    gi = table["index"].to_pylist()
    ok &= _check(gi == list(range(global_offset, global_offset + n_frames)),
                 f"global index {global_offset}..{global_offset + n_frames - 1}")

    ei = table["episode_index"].to_pylist()
    ok &= _check(all(e == ep_index for e in ei), f"episode_index all={ep_index}")

    return ok


# ── lerobot utils acceptance gate ─────────────────────────────────────────────

def check_lerobot_utils(root: Path) -> bool:
    try:
        from lerobot.datasets.utils import load_info, load_tasks, load_episodes, load_stats
    except ImportError:
        print("  skip: lerobot not importable")
        return True

    ok = True
    try:
        info = load_info(root)
        ok &= _check(info["codebase_version"] == "v3.0", "load_info() succeeds")
    except Exception as e:
        ok &= _check(False, f"load_info() raised: {e}")
        return ok

    try:
        tasks = load_tasks(root)
        ok &= _check(len(tasks) >= 1, f"load_tasks() returns {len(tasks)} task(s)")
    except Exception as e:
        ok &= _check(False, f"load_tasks() raised: {e}")

    try:
        episodes = load_episodes(root)
        ok &= _check(len(episodes) >= 1,
                     f"load_episodes() returns {len(episodes)} episode(s)")
        required = {"episode_index", "tasks", "length",
                    "data/chunk_index", "data/file_index",
                    "dataset_from_index", "dataset_to_index"}
        missing = required - set(episodes.column_names)
        ok &= _check(not missing,
                     f"load_episodes() result has routing columns (missing: {missing})")
    except Exception as e:
        ok &= _check(False, f"load_episodes() raised: {e}")

    stats = load_stats(root)
    ok &= _check(stats is not None, "load_stats() returns non-None")

    return ok


# ── test suite ────────────────────────────────────────────────────────────────

def test_collector(collector_cls, name: str, extra_parquet_cols=()):
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print("="*60)
    passed = failed = 0

    def record(result):
        nonlocal passed, failed
        if result: passed += 1
        else:      failed += 1

    fps = float(SEND_HZ)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # ── single episode ────────────────────────────────────────────────────
        print("\n-- single episode (20 frames) --")
        _run_episode(collector_cls, root, "test_task", n_frames=20)

        record(check_tasks_parquet(root))
        record(check_episodes_parquet(root, total_episodes=1, total_frames=20, fps=fps))
        record(check_info_json(root, total_ep=1, total_fr=20))
        record(check_parquet_schema(
            root / "data" / "chunk-000" / "episode_000000.parquet", extra_parquet_cols))
        record(check_parquet_values(
            root / "data" / "chunk-000" / "episode_000000.parquet", 0, 0, 20))
        record(check_timestamp_grid(
            root / "data" / "chunk-000" / "episode_000000.parquet", fps))
        record(check_stats_json(root, ["observation.state", "action", "timestamp"]))
        record(check_lerobot_utils(root))

        # ── second episode ────────────────────────────────────────────────────
        print("\n-- second episode (15 frames) --")
        _run_episode(collector_cls, root, "test_task", n_frames=15)

        record(check_episodes_parquet(root, total_episodes=2, total_frames=35, fps=fps))
        record(check_info_json(root, total_ep=2, total_fr=35))
        record(check_parquet_values(
            root / "data" / "chunk-000" / "episode_000001.parquet", 1, 20, 15))
        record(check_timestamp_grid(
            root / "data" / "chunk-000" / "episode_000001.parquet", fps))
        record(check_stats_json(root, ["observation.state", "action", "timestamp"]))

        # ── chunk rollover ────────────────────────────────────────────────────
        print("\n-- chunk rollover (3rd episode crosses chunk boundary) --")
        original_cs = collector_cls.CHUNKS_SIZE
        collector_cls.CHUNKS_SIZE = 2

        _run_episode(collector_cls, root, "test_task", n_frames=10)

        record(_check(
            (root / "videos" / "observation.images.camera" / "chunk-001" / "episode_000002.mp4").exists(),
            "chunk-001 video exists"))
        record(_check(
            (root / "data" / "chunk-001" / "episode_000002.parquet").exists(),
            "chunk-001 parquet exists"))

        collector_cls.CHUNKS_SIZE = original_cs

        # ── empty episode ─────────────────────────────────────────────────────
        print("\n-- empty episode (0 frames, should discard) --")
        empty_col = collector_cls("test_task", root / "_empty")
        empty_col.start()
        empty_col.finish(success=True)
        record(_check(empty_col.total_episodes == 0, "empty episode not counted"))

    print(f"\n{name}: {passed} passed, {failed} failed")
    return failed == 0


def main():
    from src.controller_teleop import DataCollector
    from deployment_scripts.deploy_bluerov import DAggerCollector

    all_ok = True
    all_ok &= test_collector(DataCollector,    "DataCollector (controller_teleop)")
    all_ok &= test_collector(DAggerCollector,  "DAggerCollector (deploy_bluerov)",
                              extra_parquet_cols=("is_intervention",))

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
