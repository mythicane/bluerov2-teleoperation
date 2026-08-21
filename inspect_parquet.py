import pyarrow.parquet as pq
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'grab red rod/data/chunk-000/episode_000000.parquet'
t = pq.read_table(path)
print('Schema:')
print(t.schema)
print(f'\nRows: {len(t)}')
print('\nFirst row:')
for col in t.schema.names:
    print(f'  {col}: {t[col][0]}')

obs = t['observation.state']
act = t['action']

# find rows where dive != 0
nonzero_dive = [(i, obs[i].as_py(), act[i].as_py()) for i in range(len(t)) if act[i].as_py()[2] != 0.0]
print(f'\nRows with non-zero dive: {len(nonzero_dive)}')
print(f"  {'row':>4}  {'depth':>8}  {'setpoint':>10}  {'pid_z':>8}  {'dive':>8}")
for i, o, a in nonzero_dive[:15]:
    print(f"  {i:>4}  {o[3]:>8.4f}  {o[4]:>10.4f}  {o[5]:>8.4f}  {a[2]:>8.4f}")
