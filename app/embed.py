import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.core import Segment as PSegment

from app.diarize import Segment


def _load_audio_dict(wav_path: str) -> dict:
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}

MIN_REFERENCE_VOICE_SECONDS = 3.0


@lru_cache(maxsize=1)
def _get_inference() -> Inference:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set.")
    try:
        model = Model.from_pretrained("pyannote/embedding", token=token)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "access" in msg.lower():
            raise RuntimeError(
                "pyannote/embedding access denied. Accept user conditions at "
                "huggingface.co/pyannote/embedding."
            ) from e
        raise

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    return Inference(model, window="whole")


def compute_embedding(
    wav_path: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray:
    """Returns a single speaker embedding vector for the given audio range.
    If start/end are None, embeds the entire file."""
    inference = _get_inference()
    audio = _load_audio_dict(wav_path)

    if start_sec is None and end_sec is None:
        emb = inference(audio)
    else:
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            raise ValueError("end_sec must be provided if start_sec is set")
        sample_rate = audio["sample_rate"]
        duration = audio["waveform"].shape[-1] / sample_rate
        # pyannote's internal duration check is strict; back off by one sample
        # so we never tie or exceed it due to float rounding.
        end_sec = min(end_sec, duration - 1.0 / sample_rate)
        start_sec = max(0.0, min(start_sec, end_sec))
        emb = inference.crop(audio, PSegment(start_sec, end_sec))

    return np.asarray(emb).flatten()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def assign_roles(
    segments: list[Segment],
    lecture_wav: str,
    teacher_reference_wav: str,
    threshold: float = 0.5,
) -> tuple[list[Segment], dict]:
    """Assign Teacher/Student roles by matching segment embeddings to the
    teacher reference embedding via cosine similarity."""
    if not segments:
        return segments, {"fallback_used": False, "best_similarity": 0.0}

    teacher_emb = compute_embedding(teacher_reference_wav)
    if not np.isfinite(teacher_emb).all() or np.linalg.norm(teacher_emb) == 0:
        raise ValueError(
            "Teacher reference is too short or contains no clear voice. "
            "Please upload 10-30 seconds of clear teacher audio."
        )

    longest_by_speaker: dict[str, Segment] = {}
    total_by_speaker: dict[str, float] = defaultdict(float)
    for seg in segments:
        total_by_speaker[seg.speaker_id] += seg.duration
        cur = longest_by_speaker.get(seg.speaker_id)
        if cur is None or seg.duration > cur.duration:
            longest_by_speaker[seg.speaker_id] = seg

    similarities: dict[str, float] = {}
    for speaker_id, longest in longest_by_speaker.items():
        emb = compute_embedding(lecture_wav, longest.start, longest.end)
        similarities[speaker_id] = _cosine_similarity(teacher_emb, emb)

    best_speaker = max(similarities, key=similarities.get)
    best_sim = similarities[best_speaker]

    fallback_used = False
    teacher_speaker_id = best_speaker

    if best_sim < threshold:
        fallback_used = True
        teacher_speaker_id = max(total_by_speaker, key=total_by_speaker.get)

    for seg in segments:
        seg.role = "Teacher" if seg.speaker_id == teacher_speaker_id else "Student"

    return segments, {
        "fallback_used": fallback_used,
        "best_similarity": best_sim,
        "similarities": similarities,
        "teacher_speaker_id": teacher_speaker_id,
    }
