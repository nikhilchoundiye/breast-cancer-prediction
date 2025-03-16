import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect("db.sqlite3")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Query license data (replace 'licenses' with the correct table name if different)
cursor.execute("SELECT * FROM licenses;")
print("License Records:", cursor.fetchall())

# Close connection
conn.close()
