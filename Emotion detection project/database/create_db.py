import sqlite3

conn = sqlite3.connect("emotion.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs(
id INTEGER PRIMARY KEY AUTOINCREMENT,
emotion TEXT,
gender TEXT,
date TEXT,
time TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")
