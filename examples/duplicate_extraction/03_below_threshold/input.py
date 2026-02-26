def load_users():
    db = get_db()
    return db.query("SELECT * FROM users")


def load_posts():
    db = get_db()
    return db.query("SELECT * FROM posts")
