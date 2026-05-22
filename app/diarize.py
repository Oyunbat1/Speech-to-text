import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline

MIN_SEGMENT_DURATION = 0.4
MERGE_GAP_THRESHOLD = 0.5


@dataclass
class Segment:
    start: float
    end: float
    speaker_id: str
    role: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@lru_cache(maxsize=1)
def _get_pipeline() -> Pipeline:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN not set. Create one at huggingface.co and accept the "
            "user conditions on pyannote/speaker-diarization-3.1, "
            "pyannote/segmentation-3.0, and pyannote/embedding."
        )
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "access" in msg.lower():
            raise RuntimeError(
                "pyannote model access denied. Accept the user conditions at "
                "huggingface.co/pyannote/speaker-diarization-3.1 and "
                "huggingface.co/pyannote/segmentation-3.0."
            ) from e
        raise

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def diarize(wav_path: str) -> list[Segment]:
    """Run pyannote.audio diarization. Returns segments in chronological order.
    Merges consecutive same-speaker segments <0.5s apart.
    Drops segments shorter than 0.4s."""
    pipeline = _get_pipeline()

    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    audio = {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}

    result = pipeline(audio)

    annotation = getattr(result, "exclusive_speaker_diarization", None) \
        or getattr(result, "speaker_diarization", result)

    raw: list[Segment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw.append(Segment(start=float(turn.start), end=float(turn.end), speaker_id=speaker))

    raw.sort(key=lambda s: s.start)

    merged: list[Segment] = []
    for seg in raw:
        if (
            merged
            and merged[-1].speaker_id == seg.speaker_id
            and seg.start - merged[-1].end < MERGE_GAP_THRESHOLD
        ):
            merged[-1] = Segment(
                start=merged[-1].start,
                end=max(merged[-1].end, seg.end),
                speaker_id=seg.speaker_id,
            )
        else:
            merged.append(seg)

    return [s for s in merged if s.duration >= MIN_SEGMENT_DURATION]
