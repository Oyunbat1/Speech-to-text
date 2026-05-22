import logging
import os
import uuid
import warnings
from pathlib import Path

# Load .env BEFORE any pyannote import. HF_HOME (HuggingFace cache directory)
# must be set in the environment before huggingface_hub initializes its
# default paths — otherwise model downloads go to ~/.cache/huggingface on C:.
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
logging.getLogger("pyannote").setLevel(logging.ERROR)

# torchaudio 2.11+ dropped several APIs that pyannote.audio 3.x still imports
# at module-load time: `list_audio_backends`, `AudioMetaData`, and `info`.
# Install minimal stubs BEFORE importing pyannote. Our diarize/embed paths read
# audio via soundfile, so `info` is never actually called at runtime — the
# stubs only need to make the imports succeed.
import torchaudio as _torchaudio  # noqa: E402
if not hasattr(_torchaudio, "list_audio_backends"):
    _torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(_torchaudio, "AudioMetaData"):
    from collections import namedtuple as _namedtuple

    _torchaudio.AudioMetaData = _namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )
if not hasattr(_torchaudio, "info"):
    def _torchaudio_info_stub(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "torchaudio.info is unavailable in torchaudio 2.11+. This code path "
            "shouldn't be reached because audio is read via soundfile."
        )

    _torchaudio.info = _torchaudio_info_stub

# huggingface_hub 1.x renamed the `use_auth_token` kwarg to `token` on
# hf_hub_download(), but pyannote.audio 3.x still passes the old name in
# several module-level call sites. Wrap hf_hub_download here, BEFORE pyannote
# imports its own reference, so the translation is transparent.
import huggingface_hub as _hf_hub  # noqa: E402
import inspect as _inspect  # noqa: E402
if "use_auth_token" not in _inspect.signature(_hf_hub.hf_hub_download).parameters:
    _original_hf_hub_download = _hf_hub.hf_hub_download

    def _hf_hub_download_compat(*args, **kwargs):
        if "use_auth_token" in kwargs:
            legacy = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", legacy)
        return _original_hf_hub_download(*args, **kwargs)

    _hf_hub.hf_hub_download = _hf_hub_download_compat
    # Also patch the submodule attribute so any `from huggingface_hub.file_download
    # import hf_hub_download` style import gets the wrapped version.
    _hf_hub.file_download.hf_hub_download = _hf_hub_download_compat

# PyTorch 2.6+ flipped torch.load's default `weights_only` from False to True,
# which rejects pyannote's checkpoints because they contain TorchVersion (and
# other non-allowlisted globals). pyannote 3.x's checkpoint loader passes
# `weights_only=None` through lightning_fabric, which torch 2.6+ then treats
# as True. Restore pre-2.6 behavior: when weights_only isn't explicitly True,
# load with full unpickling. Safe here because the only checkpoints loaded
# come from HuggingFace's gated pyannote repos, which the user has accepted.
import torch as _torch  # noqa: E402
_original_torch_load = _torch.load

def _torch_load_compat(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

_torch.load = _torch_load_compat

import pyannote.audio as _pyannote_audio  # noqa: E402

# pyannote.audio 4.x switched audio I/O to torchcodec, which on Windows needs
# FFmpeg's shared DLLs (avformat-*.dll, avcodec-*.dll, ...) on PATH — not just
# ffmpeg.exe. Without them, diarization crashes natively mid-pipeline with no
# Python traceback. Pin to 3.x (see requirements.txt) and fail loudly here so
# a stray upgrade is obvious.
# speechbrain (pulled in transitively by pyannote) ships a `LazyModule` whose
# `ensure_module` guard against inspect.py-triggered imports checks for a path
# ending in "/inspect.py". On Windows the separator is "\", so the guard never
# fires, and any `inspect.stack()` call from Lightning resolves LazyModule
# attributes by trying to actually import them — including the optional
# `speechbrain.integrations.k2_fsa`, which isn't installed and crashes the
# whole diarization pipeline. Make the check OS-agnostic.
try:
    import speechbrain.utils.importutils as _sb_importutils  # noqa: E402
    import importlib as _importlib  # noqa: E402
    import sys as _sys  # noqa: E402
    import inspect as _inspect_for_sb  # noqa: E402

    def _ensure_module_os_agnostic(self, stacklevel: int):
        importer_frame = None
        try:
            importer_frame = _inspect_for_sb.getframeinfo(
                _sys._getframe(stacklevel + 1)
            )
        except (AttributeError, ValueError):
            pass

        if importer_frame is not None:
            normalized = importer_frame.filename.replace("\\", "/")
            if normalized.endswith("/inspect.py"):
                raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = _importlib.import_module(self.target)
                else:
                    self.lazy_module = _importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e

        return self.lazy_module

    _sb_importutils.LazyModule.ensure_module = _ensure_module_os_agnostic
except ImportError:
    pass

_pyannote_major = int(_pyannote_audio.__version__.split(".")[0])
if _pyannote_major >= 4:
    raise RuntimeError(
        f"pyannote.audio {_pyannote_audio.__version__} is installed, but this "
        f"project requires the 3.x series. Run:\n"
        f"    pip install 'pyannote.audio>=3.1,<4' --force-reinstall\n"
        f"(4.x depends on torchcodec, which causes native crashes on Windows "
        f"unless FFmpeg shared DLLs are on PATH.)"
    )

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.pipeline import process_lecture  # noqa: E402

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
