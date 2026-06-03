import streamlit as st
import sqlite3
import bcrypt
import json

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
    # Chats table
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    title TEXT NOT NULL,
                    is_archived INTEGER DEFAULT 0)''')
    
    # برای سازگاری با دیتابیس‌های قبلی که ستون is_archived را نداشتند
    try:
        c.execute("ALTER TABLE chats ADD COLUMN is_archived INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # ستون از قبل وجود دارد

    # Messages table
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT)''')
    conn.commit()
    conn.close()

# --- User Auth Functions ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and verify_password(password, result[0]):
        return True
    return False

# --- Chat & History Functions ---
def create_new_chat(username, title="New Chat"):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (username, title, is_archived) VALUES (?, ?, 0)", (username, title))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def get_user_chats(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, title FROM chats WHERE username=? AND is_archived=0 ORDER BY id DESC", (username,))
    chats = c.fetchall()
    conn.close()
    return chats

def get_archived_chats(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, title FROM chats WHERE username=? AND is_archived=1 ORDER BY id DESC", (username,))
    chats = c.fetchall()
    conn.close()
    return chats

def update_chat_title(chat_id, title):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()

def delete_chat(chat_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()

def archive_chat(chat_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE chats SET is_archived=1 WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()

def save_message(chat_id, role, content, sources=None):
    sources_str = json.dumps(sources) if sources else "[]"
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages (chat_id, role, content, sources) VALUES (?, ?, ?, ?)", 
              (chat_id, role, content, sources_str))
    conn.commit()
    conn.close()

def get_chat_messages(chat_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT role, content, sources FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
    msgs = c.fetchall()
    conn.close()
    
    formatted_msgs = []
    for role, content, sources_str in msgs:
        formatted_msgs.append({
            "role": role,
            "content": content,
            "sources": json.loads(sources_str)
        })
    return formatted_msgs

def show_login_page():
    st.title("System Login")
    tab1, tab2 = st.tabs(["Log In", "Register"])
    
    with tab1:
        st.subheader("Log In to Your Account")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In"):
            if authenticate_user(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab2:
        st.subheader("Create a New Account")
        reg_username = st.text_input("New Username", key="reg_user")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
        if st.button("Register"):
            if reg_password != reg_password_confirm:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.warning("Password must be at least $6$ characters long.")
            else:
                if add_user(reg_username, reg_password):
                    st.success("Account successfully created! You can now log in.")
                else:
                    st.error("This username is already taken.")
