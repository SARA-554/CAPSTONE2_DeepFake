import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import joblib
from scipy.sparse import hstack, csr_matrix

MODEL_DIR = Path("models/best_text_audio_mfcc")

def load_bundle():
    vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    scaler = joblib.load(MODEL_DIR / "audio_scaler.joblib")
    clf = joblib.load(MODEL_DIR / "logreg_model.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    return vec, scaler, clf, meta

def run_ffmpeg_extract_audio(video_path: str, wav_out: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1", "-ar", "16000",
        wav_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_whisper_transcribe(wav_path: str) -> str:
    # Whisper CLI writes output files; we use temp folder and read .txt
    out_dir = Path(tempfile.mkdtemp())
    cmd = ["python3","-m","whisper", wav_path,
        "--model", "base",
        "--language", "en",
        "--task", "transcribe",
        "--output_dir", str(out_dir),
        "--output_format", "txt"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    txt = out_dir / (Path(wav_path).stem + ".txt")
    return txt.read_text(encoding="utf-8", errors="ignore").strip() if txt.exists() else ""

def mfcc_stats(wav_path: str, sr: int, n_mfcc: int) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y is None or y.size == 0:
        return np.zeros(n_mfcc * 2, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)
    return feat.astype(np.float32)

def predict_video(video_file_path: str):
    vec, scaler, clf, meta = load_bundle()
    sr = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
    inv = {int(k): v for k, v in meta["inverse_label_map"].items()}

    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
        run_ffmpeg_extract_audio(video_file_path, wav_path)

        transcript = run_whisper_transcribe(wav_path)
        X_text = vec.transform([transcript])

        x_audio = mfcc_stats(wav_path, sr=sr, n_mfcc=n_mfcc).reshape(1, -1)
        x_audio_s = scaler.transform(x_audio)

        X = hstack([X_text, csr_matrix(x_audio_s)])
        proba = clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = inv[pred_idx]

        return {
            "prediction": pred_label,
            "confidence": float(proba[pred_idx]),
            "prob_real": float(proba[0]),
            "prob_fake": float(proba[1]),
            "transcript": transcript
        }

