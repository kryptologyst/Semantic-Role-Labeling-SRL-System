import json
import sqlite3
from typing import List, Dict, Any
import os

class MockDatabase:
    """Mock database for storing sample sentences and SRL results"""
    
    def __init__(self, db_path: str = "srl_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT,
                difficulty TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS srl_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER,
                model_name TEXT,
                result_json TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences (id)
            )
        ''')
        
        # Insert sample sentences
        sample_sentences = [
            ("John gave a book to Mary on her birthday.", "simple", "easy"),
            ("The chef prepared a delicious meal for the guests.", "simple", "easy"),
            ("Sarah bought flowers from the market yesterday.", "simple", "easy"),
            ("The teacher explained the concept to the students clearly.", "simple", "medium"),
            ("After the meeting, the manager discussed the project with the team.", "complex", "medium"),
            ("The scientist discovered a new species in the Amazon rainforest.", "complex", "medium"),
            ("Despite the rain, the marathon runners continued to compete.", "complex", "hard"),
            ("The company announced its quarterly earnings during the press conference.", "complex", "hard"),
            ("The detective carefully examined the evidence at the crime scene.", "complex", "hard"),
            ("Students should complete their assignments before the deadline.", "modal", "medium"),
            ("The government implemented new policies to address climate change.", "political", "hard"),
            ("The artist painted a beautiful landscape using watercolors.", "creative", "medium"),
            ("The doctor prescribed medication to treat the patient's condition.", "medical", "medium"),
            ("The engineer designed a sustainable energy system for the building.", "technical", "hard"),
            ("The musician composed a symphony for the orchestra.", "creative", "hard"),
            ("The lawyer presented evidence to support her client's case.", "legal", "hard"),
            ("The farmer harvested crops from the field in autumn.", "agricultural", "medium"),
            ("The pilot navigated the aircraft through turbulent weather.", "aviation", "hard"),
            ("The researcher conducted experiments to test the hypothesis.", "scientific", "hard"),
            ("The athlete trained rigorously to prepare for the competition.", "sports", "medium")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO sentences (text, category, difficulty) VALUES (?, ?, ?)",
            sample_sentences
        )
        
        conn.commit()
        conn.close()
    
    def get_sentences(self, category: str = None, difficulty: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve sentences from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM sentences WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        
        query += " ORDER BY RANDOM()"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        return [dict(row) for row in rows]
    
    def save_srl_result(self, sentence_id: int, model_name: str, result: Dict[str, Any], processing_time: float):
        """Save SRL analysis result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO srl_results (sentence_id, model_name, result_json, processing_time)
            VALUES (?, ?, ?, ?)
        ''', (sentence_id, model_name, json.dumps(result), processing_time))
        
        conn.commit()
        conn.close()
    
    def get_srl_results(self, sentence_id: int = None, model_name: str = None) -> List[Dict[str, Any]]:
        """Retrieve SRL results from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT sr.*, s.text as sentence_text 
            FROM srl_results sr
            JOIN sentences s ON sr.sentence_id = s.id
            WHERE 1=1
        '''
        params = []
        
        if sentence_id:
            query += " AND sr.sentence_id = ?"
            params.append(sentence_id)
        
        if model_name:
            query += " AND sr.model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY sr.created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count sentences by category
        cursor.execute("SELECT category, COUNT(*) FROM sentences GROUP BY category")
        categories = dict(cursor.fetchall())
        
        # Count sentences by difficulty
        cursor.execute("SELECT difficulty, COUNT(*) FROM sentences GROUP BY difficulty")
        difficulties = dict(cursor.fetchall())
        
        # Count SRL results by model
        cursor.execute("SELECT model_name, COUNT(*) FROM srl_results GROUP BY model_name")
        models = dict(cursor.fetchall())
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM sentences")
        total_sentences = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM srl_results")
        total_results = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_sentences": total_sentences,
            "total_srl_results": total_results,
            "categories": categories,
            "difficulties": difficulties,
            "models": models
        }

# Sample data for testing
SAMPLE_SENTENCES = [
    "John gave a book to Mary on her birthday.",
    "The chef prepared a delicious meal for the guests.",
    "Sarah bought flowers from the market yesterday.",
    "The teacher explained the concept to the students clearly.",
    "After the meeting, the manager discussed the project with the team.",
    "The scientist discovered a new species in the Amazon rainforest.",
    "Despite the rain, the marathon runners continued to compete.",
    "The company announced its quarterly earnings during the press conference.",
    "The detective carefully examined the evidence at the crime scene.",
    "Students should complete their assignments before the deadline."
]

if __name__ == "__main__":
    # Test the mock database
    db = MockDatabase()
    
    print("Database initialized successfully!")
    print("\nSample sentences:")
    sentences = db.get_sentences(limit=5)
    for sentence in sentences:
        print(f"- {sentence['text']} ({sentence['category']}, {sentence['difficulty']})")
    
    print("\nDatabase statistics:")
    stats = db.get_statistics()
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Categories: {stats['categories']}")
    print(f"Difficulties: {stats['difficulties']}")
