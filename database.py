import sqlite3
import csv
import os
from typing import Dict, List, Tuple
import random

DB_PATH = "vocab.db"
WORDS_CSV = "words.csv"

def init_database():
    """Initialize the vocabulary database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create tables
    c.execute("""
        CREATE TABLE IF NOT EXISTS vocabulary (
            word TEXT PRIMARY KEY,
            definition TEXT,
            status TEXT DEFAULT 'unknown',
            attempts INTEGER DEFAULT 0,
            correct_attempts INTEGER DEFAULT 0,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            first_attempt BOOLEAN DEFAULT NULL,
            example_sentence TEXT
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            note_content TEXT NOT NULL,
            note_type TEXT DEFAULT 'user',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (word) REFERENCES vocabulary(word)
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            words_attempted TEXT,
            session_score REAL,
            session_data TEXT,
            session_type TEXT DEFAULT 'mixed',
            notes_created INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    
    # Add new columns to existing vocabulary table if they don't exist
    try:
        c.execute("ALTER TABLE vocabulary ADD COLUMN first_attempt BOOLEAN DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        c.execute("ALTER TABLE vocabulary ADD COLUMN example_sentence TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Seed with words if empty
    c.execute("SELECT COUNT(*) FROM vocabulary")
    if c.fetchone()[0] == 0:
        seed_vocabulary(conn)
    
    conn.close()

def seed_vocabulary(conn=None):
    """Seed the database with GRE words"""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        should_close = True
    else:
        should_close = False
    
    c = conn.cursor()
    
    try:
        with open(WORDS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c.execute("""
                    INSERT OR IGNORE INTO vocabulary (word, definition, status)
                    VALUES (?, ?, 'unknown')
                """, (row['word'].strip(), row['definition'].strip()))
        conn.commit()
        print(f"Seeded database with words from {WORDS_CSV}")
    except FileNotFoundError:
        print(f"Warning: {WORDS_CSV} not found. Database will be empty.")
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        if should_close:
            conn.close()

def get_word_status_counts() -> Dict[str, int]:
    """Get count of words by status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT status, COUNT(*) 
        FROM vocabulary 
        GROUP BY status
    """)
    
    counts = dict(c.fetchall())
    conn.close()
    
    return {
        'unknown': counts.get('unknown', 0),
        'weak': counts.get('weak', 0),
        'moderate': counts.get('moderate', 0),
        'strong': counts.get('strong', 0)
    }

def sample_words_for_quiz(count: int = 5) -> List[Tuple[str, str]]:
    """Sample words for quiz with weighted selection based on status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all words with their status
    c.execute("SELECT word, definition, status FROM vocabulary")
    all_words = c.fetchall()
    conn.close()
    
    if not all_words:
        return []
    
    # Define weights for different statuses - equal high weights for unknown/weak
    status_weights = {
        'unknown': 3,
        'weak': 3,
        'moderate': 1,
        'strong': 1
    }
    
    # Create weighted list
    weighted_words = []
    for word, definition, status in all_words:
        weight = status_weights.get(status, 1)
        weighted_words.extend([(word, definition)] * weight)
    
    # Sample without replacement
    sample_size = min(count, len(set(word for word, _ in weighted_words)))
    sampled = random.sample(weighted_words, sample_size)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sampled = []
    for word, definition in sampled:
        if word not in seen:
            unique_sampled.append((word, definition))
            seen.add(word)
    
    return unique_sampled[:count]

def get_words_by_status(status: str, limit: int = 10) -> List[Tuple[str, str]]:
    """Get words filtered by status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT word, definition 
        FROM vocabulary 
        WHERE status = ? 
        ORDER BY last_seen ASC 
        LIMIT ?
    """, (status, limit))
    
    words = c.fetchall()
    conn.close()
    return words

def update_word_status(word: str, correct: bool):
    """Update word status based on user's exact requirements"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get current status and attempts
    c.execute("""
        SELECT status, attempts, correct_attempts, first_attempt
        FROM vocabulary 
        WHERE word = ?
    """, (word,))
    
    result = c.fetchone()
    if not result:
        return
    
    current_status, attempts, correct_attempts, first_attempt = result
    new_attempts = attempts + 1
    new_correct = correct_attempts + (1 if correct else 0)
    
    # Track first attempt if this is the first time
    if first_attempt is None:
        first_attempt = correct
    
    # User's exact logic: 
    # 1st correct (unknown) -> strong
    # 1st wrong, 2nd correct -> moderate  
    # 3rd correct -> strong (else back to weak)
    if new_attempts == 1:
        new_status = 'strong' if correct else 'weak'
    elif new_attempts == 2:
        if not first_attempt and correct:
            new_status = 'moderate'
        elif correct:
            new_status = 'strong'
        else:
            new_status = 'weak'
    else:  # 3+ attempts
        if correct and current_status == 'moderate':
            new_status = 'strong'
        elif correct:
            new_status = current_status
        else:
            new_status = 'weak'
    
    # Update database
    c.execute("""
        UPDATE vocabulary 
        SET status = ?, attempts = ?, correct_attempts = ?, 
            last_seen = CURRENT_TIMESTAMP, first_attempt = ?
        WHERE word = ?
    """, (new_status, new_attempts, new_correct, first_attempt, word))
    
    conn.commit()
    conn.close()
    return new_status

def batch_update_word_statuses(word_results: Dict[str, bool]):
    """Update multiple word statuses in batch"""
    results = {}
    for word, correct in word_results.items():
        new_status = update_word_status(word, correct)
        results[word] = new_status
    return results

def save_session_data(session_data: Dict):
    """Save session data to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    import json
    
    c.execute("""
        INSERT INTO session_history (words_attempted, session_score, session_data)
        VALUES (?, ?, ?)
    """, (
        json.dumps(session_data.get('words_attempted', [])),
        session_data.get('session_score', 0.0),
        json.dumps(session_data)
    ))
    
    conn.commit()
    conn.close()

def get_word_definition(word: str) -> str:
    """Get definition for a specific word"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT definition FROM vocabulary WHERE word = ?", (word,))
    result = c.fetchone()
    conn.close()
    
    return result[0] if result else None

def search_words(query: str, limit: int = 10) -> List[Tuple[str, str]]:
    """Search words by partial match"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT word, definition 
        FROM vocabulary 
        WHERE word LIKE ? OR definition LIKE ?
        LIMIT ?
    """, (f"%{query}%", f"%{query}%", limit))
    
    words = c.fetchall()
    conn.close()
    return words

def save_note(word: str, note_content: str, note_type: str = 'user'):
    """Save a note for a word"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO user_notes (word, note_content, note_type)
        VALUES (?, ?, ?)
    """, (word, note_content, note_type))
    
    conn.commit()
    conn.close()

def get_notes_for_word(word: str) -> List[Tuple[str, str, str]]:
    """Get all notes for a specific word"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT note_content, note_type, timestamp
        FROM user_notes 
        WHERE word = ?
        ORDER BY timestamp DESC
    """, (word,))
    
    notes = c.fetchall()
    conn.close()
    return notes

def get_all_notes(limit: int = 50) -> List[Tuple[str, str, str, str]]:
    """Get all notes with word information"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT n.word, n.note_content, n.note_type, n.timestamp
        FROM user_notes n
        ORDER BY n.timestamp DESC
        LIMIT ?
    """, (limit,))
    
    notes = c.fetchall()
    conn.close()
    return notes

def batch_update_word_statuses_session_end(word_results: Dict[str, bool]):
    """Update multiple word statuses at session end"""
    results = {}
    for word, correct in word_results.items():
        new_status = update_word_status(word, correct)
        results[word] = new_status
    return results

def get_example_sentence(word: str) -> str:
    """Get example sentence for a word"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT example_sentence FROM vocabulary WHERE word = ?", (word,))
    result = c.fetchone()
    conn.close()
    
    return result[0] if result and result[0] else None

def save_example_sentence(word: str, sentence: str):
    """Save example sentence for a word"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        UPDATE vocabulary 
        SET example_sentence = ?
        WHERE word = ?
    """, (sentence, word))
    
    conn.commit()
    conn.close()

def get_recent_quiz_results(limit: int = 10) -> List[Tuple[str, bool]]:
    """Get recent quiz results for progress dots"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Simple approach: get words that were recently attempted
    c.execute("""
        SELECT word, (correct_attempts >= attempts/2) as mostly_correct
        FROM vocabulary 
        WHERE attempts > 0
        ORDER BY last_seen DESC 
        LIMIT ?
    """, (limit,))
    
    results = c.fetchall()
    conn.close()
    return results