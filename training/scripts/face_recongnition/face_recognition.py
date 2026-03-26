import os
import cv2
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort
from glob import glob
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
BASE_MODEL_DIR = os.getenv(
    "MODEL_BASE_DIR",
    os.path.join(_PROJECT_ROOT, "models"),
)
ARCFACE_ONNX = os.path.join(BASE_MODEL_DIR, "arcface.onnx")
DEFAULT_DB_PATH = os.path.join(BASE_MODEL_DIR, "arcface_db.npz")

THRESHOLD = 0.45
FACE_SIZE = (112, 112)


def l2_norm(x, epsilon=1e-10):
    return x / (np.linalg.norm(x) + epsilon)


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACE_SIZE)
    arr = face_resized.astype(np.float32)
    arr = (arr - 127.5) / 128.0
    arr = np.expand_dims(arr, axis=0)
    return arr


class ArcFaceModel:
    """
    Single model class used for:
      - building a face database in small chunks (safe on CPU machines)
      - saving/loading computed embeddings
      - performing recognition on frames
    """

    def __init__(self,
                 onnx_path: str = ARCFACE_ONNX,
                 db_path: Optional[str] = None,
                 load_db: bool = True):
        # RetinaFace is used via the static API
        self.detector = RetinaFace
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # database: mapping name -> normalized embedding (1D np.array)
        self.db: Dict[str, np.ndarray] = {}

        if db_path and load_db:
            loaded = self.load_database(db_path)
            if loaded:
                logger.info("ArcFace DB loaded from %s (%d identities)", db_path, len(self.db))
            else:
                logger.warning("No ArcFace DB at %s — starting with empty DB", db_path)

    # -------------------------
    # Embedding helpers
    # -------------------------
    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        x = preprocess_face(face)
        emb = self.session.run([self.output_name], {self.input_name: x})[0]
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb

    def get_embedding_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect largest face in frame, crop it, return L2-normalized embedding.
        Returns None if no face or on error. Does not draw on frame.
        """
        try:
            detections = self.detector.detect_faces(frame)
        except Exception:
            return None
        if not detections:
            return None
        faces = list(detections.values())
        largest = max(
            faces,
            key=lambda d: (d["facial_area"][2] - d["facial_area"][0])
            * (d["facial_area"][3] - d["facial_area"][1]),
        )
        x1, y1, x2, y2 = largest["facial_area"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face = frame[y1:y2, x1:x2]
        if face.shape[0] < 50 or face.shape[1] < 50:
            return None
        return self.get_embedding(face)

    # -------------------------
    # Database persistence
    # -------------------------
    def save_database(self, out_path: str = DEFAULT_DB_PATH):
        """
        Save current self.db to compressed npz:
        keys: names (string array), embeddings (float32 array shape (n, dim))
        """
        names = list(self.db.keys())
        if not names:
            # save an empty placeholder
            np.savez_compressed(out_path, names=np.array([]), embs=np.array([]))
            logger.info("Empty ArcFace DB saved → %s", out_path)
            return

        embs = np.stack([self.db[n] for n in names], axis=0).astype(np.float32)
        np.savez_compressed(out_path, names=np.array(names), embs=embs)
        logger.info("ArcFace DB saved: %d identities → %s", len(names), out_path)

    def load_database(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True)
        names = list(data["names"].tolist())
        embs = data["embs"]
        if embs.size == 0 or len(names) == 0:
            self.db = {}
            return True
        self.db = {n: embs[i].astype(np.float32) for i, n in enumerate(names)}
        return True

    # -------------------------
    # Chunked database builder
    # -------------------------
    def build_database_in_chunks(self,
                                 raw_database_dir: str,
                                 out_path: str = DEFAULT_DB_PATH,
                                 chunk_size: int = 10,
                                 verbose: bool = True):
        """
        Iterate over each person folder in raw_database_dir.
        For each person, process images in small chunks (chunk_size) to limit memory/CPU spikes.
        After finishing each person, compute average normalized embedding and save the DB file.
        This ensures progress is persisted frequently.
        """
        if not os.path.exists(raw_database_dir):
            raise FileNotFoundError(f"Database folder not found: {raw_database_dir}")

        persons = [
            p for p in os.listdir(raw_database_dir)
            if os.path.isdir(os.path.join(raw_database_dir, p))
        ]

        logger.info("Building ArcFace DB: %d persons in %s", len(persons), raw_database_dir)

        for idx, person in enumerate(persons, 1):
            person_path = os.path.join(raw_database_dir, person)
            images = sorted(glob(os.path.join(person_path, "*")))
            if not images:
                if verbose:
                    logger.debug("Skipping %s (no images)", person)
                continue

            embeddings: List[np.ndarray] = []
            # process in chunk_size windows
            for i in range(0, len(images), chunk_size):
                chunk = images[i:i + chunk_size]
                for img_path in chunk:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    try:
                        detections = RetinaFace.detect_faces(img)
                    except Exception:
                        continue
                    if not detections:
                        continue
                    faces = list(detections.values())
                    largest = max(
                        faces,
                        key=lambda d: (d["facial_area"][2] - d["facial_area"][0]) *
                                      (d["facial_area"][3] - d["facial_area"][1])
                    )
                    x1, y1, x2, y2 = largest["facial_area"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    face = img[y1:y2, x1:x2]
                    if face.shape[0] < 50 or face.shape[1] < 50:
                        continue
                    emb = self.get_embedding(face)
                    embeddings.append(emb)

                # after processing the chunk, you can optionally print progress
                if verbose:
                    logger.debug("[%s] processed %d/%d images", person, min(i + chunk_size, len(images)), len(images))

            if embeddings:
                avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
                avg_emb = l2_norm(avg_emb)
                self.db[person] = avg_emb
                if verbose:
                    logger.info("Trained: %s (%d samples)", person, len(embeddings))
            else:
                if verbose:
                    logger.warning("Skipped: %s (no usable faces)", person)

            # Save after each person to persist progress
            self.save_database(out_path)

        logger.info("Finished building ArcFace DB (%d identities)", len(self.db))

    # -------------------------
    # Recognition
    # -------------------------
    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        best_name = "Unknown"
        best_score = -1.0
        for name, db_emb in self.db.items():
            score = float(np.dot(embedding, db_emb))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < THRESHOLD:
            return "Unknown", best_score
        return best_name, best_score

    # -------------------------
    # Frame processing (prediction)
    # -------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            detections = RetinaFace.detect_faces(frame)
        except Exception:
            return frame

        if not detections:
            return frame

        for face_data in detections.values():
            x1, y1, x2, y2 = face_data["facial_area"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face = frame[y1:y2, x1:x2]
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue
            emb = self.get_embedding(face)
            name, score = self.recognize(emb)
            label = f"{name} | {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
