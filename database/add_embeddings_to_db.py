import cv2
import os
from collections import defaultdict
from pathlib import Path
from core.detector import detect_faces
from core.aligner import align_face
from core.embedder import embedder
from database.database import *

BASE_DIR = Path(__file__).resolve().parent.parent
CELEB_DB_PATH = BASE_DIR / "database" / "Celeb_face_embeddings.db"
DATASET_PATH = BASE_DIR / "data" / "known_faces" / "Celebrity_Faces_Dataset"


def paths_and_labels(folder_path=DATASET_PATH):
    image_paths, image_labels = [], []
    for celeb in os.listdir(folder_path):
        celeb_path = os.path.join(folder_path, celeb)
        if not os.path.isdir(celeb_path):
            continue
        for image in os.listdir(celeb_path):
            image_paths.append(os.path.join(celeb_path, image))
            image_labels.append(celeb)
    return image_paths, image_labels


def add_embeddings(folder_path=DATASET_PATH, db_path=CELEB_DB_PATH):
    db_names, _ = access_db(db_path=db_path)
    existing_names = set(db_names)

    image_paths, image_labels = paths_and_labels(folder_path)

    new_names = set(image_labels) - existing_names
    if not new_names:
        print("No new faces found.")
        return

    print(f"New faces : {new_names}")

    person_embeddings = defaultdict(list)
    for img_path, name in zip(image_paths, image_labels):
        if name not in new_names:   #only new faces
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = detect_faces(img)
        if not faces:
            continue

        bbox, lm = max(faces, key=lambda x: x[0][4])
        blob = align_face(img, lm, image_size=112)
        embedding = embedder(blob)
        person_embeddings[name].append(embedding)

    names, mean_embs = build_mean_embeddings(person_embeddings)
    conn = init_db(db_path=db_path)
    c = conn.cursor()
    for celeb_name, emb in zip(names, mean_embs):
        c.execute(
            "INSERT OR REPLACE INTO CelebFaces (name, embedding) VALUES (?, ?)",
            (celeb_name, emb.tobytes())
        )
    conn.commit()
    conn.close()
    print(f"{len(names)} New celeb added: {names}")


if __name__ == "__main__":
    add_embeddings()