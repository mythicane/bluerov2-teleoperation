"""
Depth P-controller tests.

Offline tests validate the controller math with no robot required.
Live test connects to the ROV, reads actual depth, and exercises the
setpoint + controller logic so you can confirm sensible z outputs on
a stationary robot.

Run offline:  conda run -n bluerov python unit_tests/test_depth_controller.py
Run live:     conda run -n bluerov python unit_tests/test_depth_controller.py --live
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── constants (must match controller_teleop.py) ───────────────────────────────

SEND_HZ    = 50
DEPTH_RATE = 0.5
KP_DEPTH   = 200.0
DEPTH_MIN  = -0.5
DEPTH_MAX  = 50.0


# ── controller logic (extracted for testing) ──────────────────────────────────

def update_setpoint(depth_setpoint: float, stick: float) -> float:
    """Apply one tick of stick input to the depth setpoint."""
    sp = depth_setpoint + stick * DEPTH_RATE / SEND_HZ
    return max(DEPTH_MIN, min(DEPTH_MAX, sp))


def compute_z(depth_setpoint: float, actual_depth: float) -> int:
    """P controller: returns MANUAL_CONTROL z (0-1000, 500 = hover)."""
    error = depth_setpoint - actual_depth
    return max(0, min(1000, int(500 - KP_DEPTH * error)))


# ── offline unit tests ────────────────────────────────────────────────────────

def run_offline_tests():
    passed = 0
    failed = 0

    def check(name, got, expected):
        nonlocal passed, failed
        if got == expected:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}: got {got!r}, expected {expected!r}")
            failed += 1

    def check_range(name, got, lo, hi):
        nonlocal passed, failed
        if lo <= got <= hi:
            print(f"  PASS  {name}  ({got})")
            passed += 1
        else:
            print(f"  FAIL  {name}: {got} not in [{lo}, {hi}]")
            failed += 1

    print("\n-- Setpoint update ------------------------------------------")

    sp = update_setpoint(0.0, 0.0)
    check("no stick = no setpoint change", sp, 0.0)

    # full stick down for 1 second (SEND_HZ ticks)
    sp = 0.0
    for _ in range(SEND_HZ):
        sp = update_setpoint(sp, 1.0)
    check_range("full stick down 1s moves setpoint ~0.5m", sp, 0.49, 0.51)

    # full stick up for 1 second from 0 hits the floor clamp
    sp = 0.0
    for _ in range(SEND_HZ):
        sp = update_setpoint(sp, -1.0)
    check("setpoint clamped at DEPTH_MIN", sp, DEPTH_MIN)

    # setpoint clamped at ceiling
    sp = DEPTH_MAX
    sp = update_setpoint(sp, 1.0)
    check("setpoint clamped at DEPTH_MAX", sp, DEPTH_MAX)

    print("\n-- P controller output ---------------------------------------")

    check("at setpoint -> hover (z=500)", compute_z(1.0, 1.0), 500)

    z_down = compute_z(2.0, 1.0)   # setpoint deeper than actual -> descend
    check_range("setpoint below actual -> z < 500 (descend)", z_down, 0, 499)

    z_up = compute_z(0.0, 1.0)     # setpoint shallower than actual -> ascend
    check_range("setpoint above actual -> z > 500 (ascend)", z_up, 501, 1000)

    # large error clamps output
    z_max_err = compute_z(10.0, 0.0)   # 10m deeper needed -> full descend
    check("large downward error clamps to 0", z_max_err, 0)

    z_min_err = compute_z(-0.5, 10.0)  # 10.5m up needed -> full ascend
    check("large upward error clamps to 1000", z_min_err, 1000)

    # linearity sanity: 0.5m error should give half the authority
    z_half = compute_z(1.5, 1.0)   # 0.5m deeper needed
    expected_half = max(0, min(1000, int(500 - KP_DEPTH * 0.5)))
    check("0.5m error gives expected z", z_half, expected_half)

    print(f"\n{'All' if not failed else failed} test(s) {'passed' if not failed else 'FAILED'}  "
          f"({passed} passed, {failed} failed)")
    return failed == 0


# ── live robot test ───────────────────────────────────────────────────────────

def run_live_test():
    import requests
    from config import BLUEROV_URL

    session = requests.Session()
    mav = f"{BLUEROV_URL}/mavlink2rest"

    print("\n-- Connecting to ROV ----------------------------------------")
    try:
        r = session.get(f"{mav}/mavlink/vehicles/1/components/1/messages/HEARTBEAT", timeout=3)
        assert r.ok
        print("  Connected")
    except Exception as e:
        print(f"  FAIL: cannot reach {mav}: {e}")
        return False

    # Read current depth
    try:
        p2 = session.get(
            f"{mav}/mavlink/vehicles/1/components/1/messages/SCALED_PRESSURE2",
            timeout=2).json()["message"]
        actual_depth = (p2.get("press_abs", 1013.25) - 1013.25) / 98.0
    except Exception as e:
        print(f"  FAIL: cannot read depth: {e}")
        return False

    print(f"  Actual depth: {actual_depth:.3f} m")

    print("\n-- Controller response (stationary robot) -------------------")
    print(f"  {'Scenario':<35} {'setpoint':>10} {'error':>8} {'z':>6} {'expect'}")

    scenarios = [
        ("At current depth (no error)",       actual_depth,       "z=500"),
        ("Setpoint +0.5m deeper",              actual_depth + 0.5, "z<500 (descend)"),
        ("Setpoint +1.0m deeper",              actual_depth + 1.0, "z<500 (descend)"),
        ("Setpoint -0.5m shallower",           actual_depth - 0.5, "z>500 (ascend)"),
        ("Setpoint -1.0m shallower",           actual_depth - 1.0, "z>500 (ascend)"),
        ("Large error +3m (clamp expected)",   actual_depth + 3.0, "z=0   (full descend)"),
    ]

    all_ok = True
    for label, sp, expect in scenarios:
        sp_clamped = max(DEPTH_MIN, min(DEPTH_MAX, sp))
        z = compute_z(sp_clamped, actual_depth)
        error = sp_clamped - actual_depth
        print(f"  {label:<35} {sp_clamped:>10.2f} {error:>+8.2f} {z:>6}   {expect}")

        # basic directional sanity
        if error > 0 and z >= 500:
            print(f"    FAIL: expected z < 500 for positive error")
            all_ok = False
        elif error < 0 and z <= 500:
            print(f"    FAIL: expected z > 500 for negative error")
            all_ok = False
        elif error == 0 and z != 500:
            print(f"    FAIL: expected z = 500 for zero error")
            all_ok = False

    print("\n-- Setpoint update simulation (10 ticks full stick down) ----")
    sp = actual_depth
    for i in range(10):
        sp = update_setpoint(sp, 1.0)
    delta = sp - actual_depth
    expected_delta = DEPTH_RATE / SEND_HZ * 10
    ok = abs(delta - expected_delta) < 0.0001
    print(f"  After 10 ticks full-down: setpoint moved {delta:.4f}m "
          f"(expected {expected_delta:.4f}m)  {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

    print(f"\n{'PASS' if all_ok else 'FAIL'}  live controller test")
    return all_ok


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Also run the live ROV test (robot must be connected)")
    args = ap.parse_args()

    ok = run_offline_tests()

    if args.live:
        ok = run_live_test() and ok

    sys.exit(0 if ok else 1)
