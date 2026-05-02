import sqlite3
conn = sqlite3.connect('backend/aquaai.db')
c = conn.cursor()

print("=== RAINFALL ===")
c.execute("SELECT model_name, accuracy, f1 FROM model_metrics WHERE module='rainfall' ORDER BY nse DESC")
for r in c.fetchall():
    print(f"  {r[0]:20s} acc={r[1]}  f1={r[2]}")

print("\n=== TANK ===")
c.execute("SELECT model_name, accuracy, f1 FROM model_metrics WHERE module='tank' ORDER BY accuracy DESC")
for r in c.fetchall():
    print(f"  {r[0]:20s} acc={r[1]}  f1={r[2]}")

print("\n=== IRRIGATION ===")
c.execute("SELECT model_name, accuracy, f1 FROM model_metrics WHERE module='irrigation' ORDER BY accuracy DESC")
for r in c.fetchall():
    print(f"  {r[0]:20s} acc={r[1]}  f1={r[2]}")

conn.close()
