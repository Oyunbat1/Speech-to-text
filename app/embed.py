import os
import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.core import Segment as PSegment

from app.diarize import Segment

# pyannote's StatsPool computes std with Bessel's correction. On very short
# inputs the post-conv time dimension can collapse to <=1 frame, producing a
# noisy UserWarning + NaN. We pad short crops to MIN_EMBEDDING_DURATION below,
# and silence the residual warning (NaN is caught downstream by the
# isfinite() check in _cosine_similarity).
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)


def _load_audio_dict(wav_path: str) -> dict:
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}

MIN_REFERENCE_VOICE_SECONDS = 3.0
# pyannote/embedding's pooling layer computes std with Bessel's correction.
# If the post-conv time dimension collapses to <=1 frame, std() returns NaN.
# Empirically ~2s of input audio is enough to keep that dimension >=2 frames.
MIN_EMBEDDING_DURATION = 2.0


@lru_cache(maxsize=1)
def _get_inference() -> Inference:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set.")
    try:
        model = Model.from_pretrained("pyannote/embedding")
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
    sample_rate = audio["sample_rate"]
    duration = audio["waveform"].shape[-1] / sample_rate

    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = duration

    # Pad short ranges up to MIN_EMBEDDING_DURATION using surrounding audio.
    # Without this, short diarization segments (or short teacher reference
    # clips) make pyannote's pooling layer compute std() on a single frame,
    # producing a UserWarning + NaN embedding.
    if end_sec - start_sec < MIN_EMBEDDING_DURATION:
        pad = (MIN_EMBEDDING_DURATION - (end_sec - start_sec)) / 2
        start_sec = max(0.0, start_sec - pad)
        end_sec = min(duration, end_sec + pad)
        shortfall = MIN_EMBEDDING_DURATION - (end_sec - start_sec)
        if shortfall > 0:
            if start_sec > 0:
                start_sec = max(0.0, start_sec - shortfall)
            else:
                end_sec = min(duration, end_sec + shortfall)

    # pyannote's internal duration check is strict; back off by one sample
    # so we never tie or exceed it due to float rounding.
    end_sec = min(end_sec, duration - 1.0 / sample_rate)
    start_sec = max(0.0, min(start_sec, end_sec))
    emb = inference.crop(audio, PSegment(start_sec, end_sec))

    return np.asarray(emb).flatten()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
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

    # Number non-teacher speakers as "Student 1", "Student 2", ... in
    # chronological order of first appearance.
    first_seen: dict[str, float] = {}
    for seg in segments:
        if seg.speaker_id == teacher_speaker_id:
            continue
        if seg.speaker_id not in first_seen:
            first_seen[seg.speaker_id] = seg.start
    student_order = sorted(first_seen, key=first_seen.get)
    student_label: dict[str, str] = {
        sid: f"Student {i + 1}" for i, sid in enumerate(student_order)
    }

    for seg in segments:
        if seg.speaker_id == teacher_speaker_id:
            seg.role = "Teacher"
        else:
            seg.role = student_label[seg.speaker_id]

    return segments, {
        "fallback_used": fallback_used,
        "best_similarity": best_sim,
        "similarities": similarities,
        "teacher_speaker_id": teacher_speaker_id,
        "student_labels": student_label,
    }
