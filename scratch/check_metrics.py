import sqlite3
import pandas as pd

conn = sqlite3.connect('backend/aquaai.db')
df = pd.read_sql_query("SELECT module, model_name, accuracy, f1 FROM model_metrics WHERE module = 'irrigation' ORDER BY accuracy DESC", conn)
print(df)
conn.close()
