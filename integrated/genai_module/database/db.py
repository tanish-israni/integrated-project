import hashlib
import sqlite3
from collections import Counter

from config.config import DB_PATH, STORAGE_DIR


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db() -> None:
    """
    Create the SQLite database and required tables on first run.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        # Users are stored once and linked to many saved generations.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Each generation belongs to one user and stores the saved image path.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        connection.commit()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """
    Create a new user account.
    """
    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
                """,
                (username.strip(), _hash_password(password)),
            )
            connection.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def authenticate_user(username: str, password: str) -> tuple[bool, dict | None]:
    """
    Validate login credentials and return the user record if found.
    """
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, username
            FROM users
            WHERE username = ? AND password_hash = ?
            """,
            (username.strip(), _hash_password(password)),
        )
        row = cursor.fetchone()

    if not row:
        return False, None

    return True, {"id": row[0], "username": row[1]}


def save_generation(user_id: int, prompt: str, generated_text: str, image_path: str) -> None:
    """
    Save a generated hospitality concept for a specific user.
    """
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO generations (user_id, prompt, generated_text, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, prompt, generated_text, image_path),
        )
        connection.commit()


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
    return [token for token in cleaned.split() if len(token) > 2]


def _similarity_score(query: str, candidate: str) -> int:
    query_tokens = Counter(_tokenize(query))
    candidate_tokens = Counter(_tokenize(candidate))
    return sum(min(query_tokens[token], candidate_tokens[token]) for token in query_tokens)


def get_relevant_history(user_id: int, prompt: str, limit: int = 3) -> list[dict]:
    """
    Retrieve the most relevant previous generations for one user.
    This acts as a lightweight personal RAG layer over saved history.
    """
    records = get_user_generations(user_id)
    scored_records = []

    for record in records:
        combined_text = f"{record['prompt']} {record['generated_text']}"
        score = _similarity_score(prompt, combined_text)
        if score > 0:
            scored_records.append((score, record))

    scored_records.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
    return [record for _, record in scored_records[:limit]]


def get_user_generations(user_id: int) -> list[dict]:
    """
    Return saved generations for one user, newest first.
    """
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, prompt, generated_text, image_path, created_at
            FROM generations
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "prompt": row[1],
            "generated_text": row[2],
            "image_path": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]
