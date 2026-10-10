# region step 1 : libraries
import os
import sqlite3
import numpy as np
from pathlib import Path
# endregion

#region variables
BASE_DIR = Path(__file__).resolve().parent.parent
CELEB_DB_PATH = BASE_DIR / "database" / "CelebEmbedds_edgecv.db"
PRESENCE_DB_PATH = BASE_DIR / "database" / "PresenceFaces.db"
DATASET_PATH = BASE_DIR / "data" / "known_faces" / "Celebrity_Faces_Dataset"
# endregion

# region step 2 : loading dataset
image_labels = []
image_paths = []
for celeb in os.listdir(DATASET_PATH):
    celeb_path = os.path.join(DATASET_PATH, celeb)
    for image in os.listdir(celeb_path):
        img_path = os.path.join(celeb_path, image)
        image_paths.append(img_path)
        image_labels.append(celeb)
# endregion

# region step 3 : functions
def init_db(db_path=CELEB_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CelebFaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn

def presence_db(db_path=PRESENCE_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS PresenceFaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                score FLOAT,
                date DATETIME
            )
        """)
    conn.commit()
    return conn

def build_mean_embeddings(embeddings_dict: dict):
    """
    Input:
        embeddings_dict: dict
            {
              "Angelina Jolie": [emb1, emb2, ..., embN],
              "Brad Pitt":      [emb1, emb2, ..., embN],
            }
    Output:
        names: list[str]
        mean_embeddings: list[np.ndarray]
    """
    celebrity_names = []
    mean_embeddings = []

    for clb_name, embs in embeddings_dict.items():
        if len(embs) == 0:
            continue

        embs = np.stack(embs)              # (N, 512)
        mean_emb = np.mean(embs, axis=0)   # (512,)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        celebrity_names.append(clb_name)
        mean_embeddings.append(mean_emb.astype(np.float32))

    return celebrity_names, mean_embeddings

def add_to_db(conn, names, mean_embs):
    cursor = conn.cursor()
    for celeb_name, emb in zip(names, mean_embs):
        emb_blob = emb.tobytes()
        cursor.execute(
            "INSERT OR REPLACE INTO CelebFaces (name, embedding) VALUES (?, ?)",
            (celeb_name, emb_blob) )
    conn.commit()

def add_to_presence_db(conn, celeb_name, score, date):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO PresenceFaces (name, score, date) VALUES (?, ?, ?)",
        (celeb_name, score, date) )
    conn.commit()

def access_db(db_path=CELEB_DB_PATH):
    """
    :return:
        db_names: list[str]
        db_embeddings: np.ndarray (N, 512)
    """
    db_names = []
    db_embeddings = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM CelebFaces")
        rows = cursor.fetchall()

    for name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)

        if emb.size != 512:
            continue

        db_names.append(name)
        db_embeddings.append(emb)

    if len(db_embeddings) == 0:
        return [], np.empty((0, 512), dtype=np.float32)

    return db_names, np.vstack(db_embeddings)

# endregion



#db_names, db_embeddings = access_db()
#print(db_embeddings.shape) # shape: (N, 512)
