import av
# Confirms whether this PyAV build supports UDP (it doesn't on this machine — expected)
print("udp" in av.library_versions)
try:
    container = av.open("udp://0.0.0.0:5600")
except Exception as e:
    print(f"Result: {e}")
