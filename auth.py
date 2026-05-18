import sqlite3
import bcrypt

# Create DB
def create_users_table():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


# Signup
def signup_user(username, email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(
    password.encode(),
    bcrypt.gensalt()
).decode()

    try:
        cursor.execute("""
        INSERT INTO users (username, email, password)
        VALUES (?, ?, ?)
        """, (username, email, hashed_password))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()

# Login
def login_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM users
    WHERE username=?
    """, (username,))

    user = cursor.fetchone()

    conn.close()

    if user:
        stored_password = user[3]

        # Convert to bytes if needed
        if isinstance(stored_password, str):
            stored_password = stored_password.encode()

        if bcrypt.checkpw(
            password.encode(),
            stored_password
        ):
            return True

    return False