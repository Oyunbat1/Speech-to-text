import io
import subprocess
from pathlib import Path

from pydub import AudioSegment


def extract_audio(media_path: str, out_wav_path: str) -> None:
    """ffmpeg: video or audio file -> mono 16kHz PCM WAV. Raises on failure."""
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(media_path),
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-vn",
        str(out_wav_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (apt install ffmpeg / brew install ffmpeg)."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed for {media_path}: {stderr}") from e

    if not Path(out_wav_path).exists():
        raise RuntimeError(f"ffmpeg did not produce {out_wav_path}")


def slice_audio(wav_path: str, start_sec: float, end_sec: float) -> bytes:
    """Returns WAV bytes for the given time range. Used to send to Chimege."""
    audio = AudioSegment.from_wav(wav_path)
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    clip = audio[start_ms:end_ms]

    buf = io.BytesIO()
    clip.export(buf, format="wav")
    return buf.getvalue()
