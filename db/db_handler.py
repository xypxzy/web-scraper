import sqlite3
import json
from datetime import datetime


class DBHandler:
    def __init__(self, db_name='analysis_results'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        analysis_date TIMESTAMP NOT NULL,
                        recommendation TEXT NOT NULL,
                        elements TEXT NOT NULL
                    )
                """)
        self.conn.commit()

    def save_analysis(self, url, recommendations, elements):
        self.cursor.execute("""
                    INSERT INTO recommendations (url, analysis_date, recommendation, elements)
                    VALUES (?, ?, ?, ?)
                """, (url, datetime.now(), json.dumps(recommendations), json.dumps(elements)))
        self.conn.commit()

    def close(self):
        self.conn.close()
