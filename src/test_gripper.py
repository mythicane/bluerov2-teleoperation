import requests, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BLUEROV_URL

MAV = f"{BLUEROV_URL}/mavlink2rest"
HEADER = {"system_id": 255, "component_id": 0, "sequence": 0}
s = requests.Session()

def post(msg):
    return s.post(f"{MAV}/mavlink", json={"header": HEADER, "message": msg}, timeout=1)

def param_id_chars(name):
    return list(name.ljust(16, chr(0))[:16])

def set_param(name, value):
    r = post({"type": "PARAM_SET", "target_system": 1, "target_component": 1,
              "param_id": param_id_chars(name), "param_value": float(value),
              "param_type": {"type": "MAV_PARAM_TYPE_INT8"}})
    print(f"  PARAM_SET {name}={value}: {r.status_code} {r.text[:60]}")
    time.sleep(0.5)

def servo_raw():
    r = s.get(f"{MAV}/mavlink/vehicles/1/components/1/messages/SERVO_OUTPUT_RAW", timeout=1)
    val = r.json()["message"]["servo11_raw"]
    print(f"  servo11 = {val}")
    return val

def do_set_servo(channel, pwm):
    return post({"type": "COMMAND_LONG", "target_system": 1, "target_component": 1,
                 "command": {"type": "MAV_CMD_DO_SET_SERVO"}, "confirmation": 0,
                 "param1": float(channel), "param2": float(pwm),
                 "param3": 0, "param4": 0, "param5": 0, "param6": 0, "param7": 0})

# Test A: SET_ACTUATOR_CONTROL_TARGET
print("=== Test A: SET_ACTUATOR_CONTROL_TARGET group=1 ===")
r = post({"type": "SET_ACTUATOR_CONTROL_TARGET", "time_usec": 0,
          "group_mlx": 1, "target_system": 1, "target_component": 1,
          "controls": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
print(f"  POST: {r.status_code} {r.text[:80]}")
time.sleep(0.5)
servo_raw()

# Test B: Disable SERVO11_FUNCTION, use DO_SET_SERVO, restore
print("\n=== Test B: Disable SERVO11_FUNCTION -> DO_SET_SERVO -> restore ===")
set_param("SERVO11_FUNCTION", 0)
time.sleep(0.3)

print("  Opening (1900)...")
r = do_set_servo(11, 1900)
ack = s.get(f"{MAV}/mavlink/vehicles/1/components/1/messages/COMMAND_ACK", timeout=1)
ack_result = ack.json()["message"]["result"]["type"] if ack.ok else "n/a"
print(f"  ACK: {ack_result}")
time.sleep(0.5)
servo_raw()

time.sleep(1.5)

print("  Closing (1100)...")
do_set_servo(11, 1100)
time.sleep(0.5)
servo_raw()

print("\n=== Restoring SERVO11_FUNCTION = 184 ===")
set_param("SERVO11_FUNCTION", 184)
servo_raw()
