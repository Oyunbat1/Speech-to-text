import logging
import os
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
logging.getLogger("pyannote").setLevel(logging.ERROR)

import pyannote.audio as _pyannote_audio

# pyannote.audio 4.x switched audio I/O to torchcodec, which on Windows needs
# FFmpeg's shared DLLs (avformat-*.dll, avcodec-*.dll, ...) on PATH — not just
# ffmpeg.exe. Without them, diarization crashes natively mid-pipeline with no
# Python traceback. Pin to 3.x (see requirements.txt) and fail loudly here so
# a stray upgrade is obvious.
_pyannote_major = int(_pyannote_audio.__version__.split(".")[0])
if _pyannote_major >= 4:
    raise RuntimeError(
        f"pyannote.audio {_pyannote_audio.__version__} is installed, but this "
        f"project requires the 3.x series. Run:\n"
        f"    pip install 'pyannote.audio>=3.1,<4' --force-reinstall\n"
        f"(4.x depends on torchcodec, which causes native crashes on Windows "
        f"unless FFmpeg shared DLLs are on PATH.)"
    )

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import process_lecture

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./tmp/outputs"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4a", ".mp3", ".wav"}

app = FastAPI(title="Mongolian Lecture Transcriber")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(500, "index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return suffix


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"File too large. Max upload is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
                )
            f.write(chunk)


def _cleanup_files(*paths: Path) -> None:
    for p in paths:
        try:
            if p and p.exists():
                p.unlink()
        except OSError:
            pass


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    lecture: UploadFile = File(...),
    teacher_reference: UploadFile = File(...),
) -> FileResponse:
    job_id = uuid.uuid4().hex[:12]

    lecture_suffix = _safe_suffix(lecture.filename or "")
    teacher_suffix = _safe_suffix(teacher_reference.filename or "")

    lecture_path = UPLOAD_DIR / f"{job_id}__lecture{lecture_suffix}"
    teacher_path = UPLOAD_DIR / f"{job_id}__teacher{teacher_suffix}"
    output_path = OUTPUT_DIR / f"{job_id}__transcript.xlsx"

    try:
        await _save_upload(lecture, lecture_path)
        await _save_upload(teacher_reference, teacher_path)

        process_lecture(
            lecture_path=str(lecture_path),
            teacher_reference_path=str(teacher_path),
            output_xlsx_path=str(output_path),
        )
    except HTTPException:
        _cleanup_files(lecture_path, teacher_path, output_path)
        raise
    except ValueError as e:
        _cleanup_files(lecture_path, teacher_path, output_path)
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        _cleanup_files(lecture_path, teacher_path, output_path)
        raise HTTPException(500, str(e)) from e
    except Exception as e:
        _cleanup_files(lecture_path, teacher_path, output_path)
        raise HTTPException(500, f"Pipeline failed: {e}") from e

    background_tasks.add_task(_cleanup_files, lecture_path, teacher_path, output_path)

    return FileResponse(
        path=str(output_path),
        filename="transcript.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        background=background_tasks,
    )
