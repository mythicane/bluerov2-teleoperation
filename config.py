from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Root for LeRobot-format task datasets recorded by src/controller_teleop.py,
# e.g. REPO_ROOT/"grab red rod"/{data,meta,videos}. Point this at wherever you
# want recordings to live -- defaults to the repo root itself, matching the
# empty "grab <color> rod[ under currents]" folders checked into this repo.
DATA_ROOT = REPO_ROOT

# Directory of QGroundControl .tlog files, used by src/explore.py. Point this
# at your own QGroundControl install (Settings > General > File Save Path).
TLOG_DIR = Path.home() / "Documents" / "QGroundControl" / "Telemetry"

# BlueROV2 companion computer (BlueOS) — reachable over the Fathom-X tether at
# its default static IP. Change if your vehicle uses a different address.
BLUEROV_URL = "http://192.168.2.2"
