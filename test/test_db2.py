import os
import psycopg2
from config import settings

uri = settings.supabase_uri.replace("postgresql+asyncpg://", "postgresql://")
conn = psycopg2.connect(uri)
cur = conn.cursor()
cur.execute("SELECT session_id, length(user_identifier), length(session_data::text) FROM user_sessions ORDER BY length(user_identifier) DESC LIMIT 5;")
print(cur.fetchall())
