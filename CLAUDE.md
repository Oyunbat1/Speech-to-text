# CLAUDE.md

This file gives AI coding assistants (Claude Code, Cursor, etc.) the context needed to work effectively on this project. **Read this before writing any code.**

---

## 1. Project Overview

**Name:** Mongolian Lecture Transcriber (working title)
**Type:** Single-page web app (MVP, no auth, no DB)
**Goal:** User uploads (a) a short teacher voice reference clip and (b) a lecture video → receives an Excel spreadsheet with the lecture transcribed in Mongolian, with each row labeled as **Teacher** or **Student**.

### What the user does
1. Upload a **teacher voice reference** (10–30 second audio clip of just the teacher speaking).
2. Upload the **lecture video** (mp4, mov, mkv, m4a, mp3, wav).
3. Wait for processing (progress shown in UI).
4. Download a `.xlsx` file: columns = `Start | End | Role | Text`.

### What the system does internally
```
teacher reference clip  →  audio (WAV 16kHz mono)
                        →  speaker embedding vector  ─┐
                                                      │
lecture video           →  audio (WAV 16kHz mono)     │
                        →  speaker diarization (pyannote)
                        →  speaker embedding per detected speaker
                        →  cosine-similarity match vs. reference ◀┘
                        →  closest match = "Teacher", others = "Student"
                        →  per-segment Mongolian STT (Chimege API)
                        →  assemble rows
                        →  write .xlsx
                        →  return download
```

---

## 2. Scope (MVP — TODAY)

### In scope
- Single-user, single-upload-at-a-time processing
- Two file inputs: teacher voice reference + lecture video
- Synchronous request (UI shows "processing..." spinner; OK to wait)
- Local file storage in a temp directory; files deleted after download
- Reference-voice-based teacher identification with longest-speaker fallback
- Mongolian language only

### Explicitly out of scope (do NOT add)
- Authentication, user accounts, sessions
- Database (no Postgres, no SQLite, no ORM)
- Payment integration
- Multi-language support
- Identifying *individual* students (all non-teachers are just "Student")
- Async job queue (Celery, RQ) — synchronous is fine for MVP
- Docker / deployment — runs locally for now
- Tests beyond a single smoke test on the pipeline
- Frontend framework (no React, Vue, Next) — plain HTML + vanilla JS only

If a task seems to require any of the above, **stop and confirm with the user first**.

---

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Best ecosystem for audio/ML |
| Web framework | FastAPI | Fast, async-friendly, auto OpenAPI |
| Server | Uvicorn | Standard for FastAPI |
| Audio extraction | ffmpeg (CLI, via subprocess) | Universal, handles all formats |
| Audio slicing | pydub | Simple Python API on top of ffmpeg |
| Speaker diarization | pyannote.audio 3.x | Open source, state-of-the-art |
| Speaker embeddings | pyannote/embedding | Matches segments to teacher reference |
| Speech-to-text | **Chimege API** (Mongolian) | Only viable Mongolian STT; ~96% accuracy claim |
| Spreadsheet output | openpyxl | Native .xlsx, no Excel install needed |
| Frontend | Plain HTML + vanilla JS | Zero build step |
| Config | `.env` via `python-dotenv` | Keep API tokens out of code |

### Required Python packages (`requirements.txt`)
```
fastapi
uvicorn[standard]
python-multipart
pyannote.audio
pydub
openpyxl
requests
python-dotenv
torch
torchaudio
numpy
```

### System dependencies
- `ffmpeg` must be installed and on PATH (`apt install ffmpeg` / `brew install ffmpeg`)

---

## 4. External Services & Secrets

### Chimege API (Mongolian STT)
- Docs: `https://docs.api.chimege.com/v1.2/en/`
- Marketing/signup: `https://chimege.com/`
- Auth: token-based — set `CHIMEGE_TOKEN` in `.env`
- Send audio (WAV/MP3) as request body; receive Mongolian text
- **The exact endpoint URL, header name, and response schema must be read from the docs page in the browser** when the user signs up — the docs render client-side and aren't scrapeable. Confirm with user / fetch from their account before coding the client.

### HuggingFace (for pyannote models)
- Required steps **before any code runs**:
  1. Create account at `huggingface.co`
  2. Accept user conditions on:
     - `huggingface.co/pyannote/speaker-diarization-3.1`
     - `huggingface.co/pyannote/segmentation-3.0`
     - `huggingface.co/pyannote/embedding`  ← **new, for the reference-voice matching**
  3. Create access token → set `HF_TOKEN` in `.env`
- Without this, pyannote will throw an auth error on first run.

### `.env` template
```env
CHIMEGE_TOKEN=your_chimege_api_token_here
HF_TOKEN=your_huggingface_token_here
UPLOAD_DIR=./tmp/uploads
OUTPUT_DIR=./tmp/outputs
TEACHER_MATCH_THRESHOLD=0.5
```

`TEACHER_MATCH_THRESHOLD` is the minimum cosine similarity to confidently identify a speaker as the teacher. Below this, the system falls back to "longest-speaking = teacher" and flags the output.

---

## 5. Project Structure

```
.
├── CLAUDE.md                # this file
├── README.md                # user-facing setup instructions
├── requirements.txt
├── .env                     # gitignored; secrets
├── .env.example             # committed; template
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes
│   ├── pipeline.py          # orchestrates the full flow
│   ├── audio.py             # extract_audio, slice_audio (ffmpeg/pydub)
│   ├── diarize.py           # pyannote wrapper
│   ├── embed.py             # speaker embeddings + reference matching
│   ├── transcribe.py        # Chimege API client
│   └── sheet.py             # openpyxl xlsx writer
├── static/
│   └── index.html           # single-page UI (two uploads + status + download)
└── tmp/
    ├── uploads/             # incoming files (gitignored)
    └── outputs/             # generated xlsx (gitignored)
```

Keep modules small and pure. `pipeline.py` is the only place that calls multiple modules together.

---

## 6. Module Contracts

### `app/audio.py`
```python
def extract_audio(media_path: str, out_wav_path: str) -> None:
    """ffmpeg: video or audio file → mono 16kHz PCM WAV. Raises on failure.
    Works for both the teacher reference clip and the lecture video."""

def slice_audio(wav_path: str, start_sec: float, end_sec: float) -> bytes:
    """Returns WAV bytes for the given time range. Used to send to Chimege."""
```

### `app/diarize.py`
```python
@dataclass
class Segment:
    start: float           # seconds
    end: float             # seconds
    speaker_id: str        # raw label from pyannote, e.g. "SPEAKER_00"
    role: str | None = None  # "Teacher" or "Student", filled later

def diarize(wav_path: str) -> list[Segment]:
    """Run pyannote.audio diarization. Returns segments in chronological order.
    Merges consecutive same-speaker segments <0.5s apart.
    Drops segments shorter than 0.4s."""
```

### `app/embed.py`  *(new module — reference-voice matching)*
```python
import numpy as np

def compute_embedding(wav_path: str,
                      start_sec: float | None = None,
                      end_sec: float | None = None) -> np.ndarray:
    """Returns a single speaker embedding vector for the given audio range.
    If start/end are None, embeds the entire file (used for the teacher reference).
    Uses pyannote/embedding model."""

def assign_roles(segments: list[Segment],
                 lecture_wav: str,
                 teacher_reference_wav: str,
                 threshold: float = 0.5) -> tuple[list[Segment], dict]:
    """For each unique speaker_id in segments:
        1. Pick the longest segment for that speaker.
        2. Compute embedding for that segment from lecture_wav.
        3. Cosine-similarity to teacher reference embedding.
       The speaker with the highest similarity ABOVE threshold → 'Teacher'.
       Everyone else → 'Student'.

       If NO speaker is above threshold, fall back to:
         longest-total-airtime speaker = Teacher, others = Student.
       Set fallback flag in returned info dict.

       Returns (mutated_segments, {'fallback_used': bool, 'best_similarity': float})."""
```

### `app/transcribe.py`
```python
def transcribe(wav_bytes: bytes) -> str:
    """POST WAV bytes to Chimege STT. Returns Mongolian transcript text.
    Retries up to 3x on transient HTTP errors. Returns '' if all retries fail
    (do NOT raise — one bad segment shouldn't kill the whole lecture)."""
```

### `app/sheet.py`
```python
def write_xlsx(segments: list[Segment],
               texts: list[str],
               out_path: str,
               fallback_warning: bool = False) -> None:
    """Columns: Start (HH:MM:SS) | End (HH:MM:SS) | Role | Text.
    Auto-width columns. Bold header row.
    If fallback_warning=True, add a yellow-highlighted note row at the top:
      '⚠️ Teacher voice not confidently matched — roles assigned by speaking time.'"""
```

### `app/pipeline.py`
```python
def process_lecture(lecture_path: str,
                    teacher_reference_path: str,
                    output_xlsx_path: str,
                    progress_cb: Callable[[str], None] | None = None) -> None:
    """End-to-end:
       1. extract_audio(lecture)       → lecture.wav
       2. extract_audio(teacher_ref)   → teacher.wav
       3. diarize(lecture.wav)         → segments
       4. assign_roles(segments, lecture.wav, teacher.wav) → roles + info
       5. transcribe each segment via Chimege
       6. write_xlsx(...)
       progress_cb is called with human-readable strings at each phase."""
```

### `app/main.py`
- `GET /` → serves `static/index.html`
- `POST /upload` → multipart form with **two files** (`lecture` + `teacher_reference`) → runs pipeline → returns `.xlsx` as `FileResponse`
- Cleanup: delete uploaded files and intermediate WAVs after response (use `BackgroundTasks`)

---

## 7. Key Implementation Notes

### Audio format
Both the teacher reference and the lecture are converted to **16kHz mono WAV** once at the start. All downstream steps (diarization, embedding, slicing for STT) use these WAV files. No per-step resampling.

### Teacher reference clip guidelines (show in UI)
- **10–30 seconds** is the sweet spot. Too short = unreliable embedding; too long = wastes upload time.
- Should be **only the teacher speaking**, no overlapping voices, minimal background noise.
- Same recording environment as the lecture is ideal (same room/mic) but not required.

### Why "longest segment per speaker" for the embedding match
Computing one embedding per speaker (using their longest contiguous segment) is far cheaper than embedding every segment, and it's more accurate because longer audio gives more stable embeddings. Don't average embeddings across all segments of a speaker — slower and not meaningfully better.

### Cosine similarity threshold
Default `0.5`. Tune by testing. Pyannote embeddings are L2-normalized so cosine similarity ∈ [-1, 1]; same-speaker matches typically score 0.6–0.9, different speakers 0.0–0.3. **0.5 is the safe middle.**

### Fallback behavior (critical for UX)
If no detected speaker exceeds the threshold:
- Likely cause: (a) teacher's voice not actually in the lecture, (b) reference clip too noisy, (c) teacher's voice changed (cold, different mic).
- Don't fail — fall back to longest-airtime = teacher and **flag it in the output sheet**.
- Log the best similarity score for debugging.

### Skip silence / very short segments
Drop segments shorter than 0.4 seconds before sending to Chimege — usually breath/cough noise, wastes API quota.

### Progress reporting
For MVP, log to stdout. **Don't add SSE/WebSocket.** A spinner with no progress text is acceptable for v1.

### Error handling philosophy
- Chimege failure on one segment → empty text in that row, continue.
- ffmpeg failure → fatal, return HTTP 500 with clear message.
- pyannote failure → fatal, but check the message: "401" / "gated" means HF token missing or licenses not accepted.
- Reference clip too short (<3s of voice detected) → return HTTP 400 with friendly message: *"Teacher reference is too short or contains no clear voice. Please upload 10–30 seconds of clear teacher audio."*

---

## 8. Performance Expectations

| Lecture length | CPU-only processing time (rough) |
|---|---|
| 5 min | 3–6 min |
| 30 min | 25–35 min |
| 60 min | 45–70 min |

Reference embedding adds ~5 seconds total. Per-speaker embeddings add ~2–5 seconds. Negligible compared to diarization + STT.

Most time is in (1) pyannote diarization, (2) Chimege API calls. **Do not parallelize Chimege calls in MVP** — be polite to the API, avoid rate limits, debug sequentially.

GPU (CUDA) gives 5–10x speedup on diarization and embeddings. Document this in README.

---

## 9. Testing Strategy

For MVP, **one end-to-end smoke test** is enough:
1. Have a 30-second test clip with two clearly distinct voices in `tests/fixtures/sample.mp4`
2. Have a 10-second teacher voice clip in `tests/fixtures/teacher_ref.wav`
3. `tests/test_smoke.py` runs `process_lecture()` and asserts:
   - Output xlsx exists
   - Has > 0 data rows
   - Has at least one row with role="Teacher"
   - Has at least one row with role="Student"
   - The Teacher row(s) correspond to the speaker actually in the reference (manually verify once)

No unit tests for individual modules unless something breaks repeatedly.

---

## 10. Known Limitations (document in README)

- Requires a clean teacher voice sample (10–30s, no overlapping voices).
- Only Mongolian. English / code-switching may produce garbage.
- No background noise reduction — noisy classroom recordings will degrade STT and embedding accuracy.
- Cannot distinguish individual students — all non-teachers are grouped as "Student".
- Synchronous request: large files may hit browser/proxy timeouts. Cap upload size at ~200MB in `main.py`.
- Each Chimege API call costs money. No caching.

---

## 11. Things to Ask the User If Unclear

- The exact Chimege endpoint URL and request format (from their account dashboard).
- Max acceptable file size for both teacher reference and lecture.
- Whether the sheet should have a summary row (total minutes per role) — **not in spec, don't add**.
- Whether to support multiple teacher references (e.g., two co-teachers) — **not in spec, single teacher only**.

---

## 12. Anti-patterns — DO NOT do these

- ❌ Don't use Whisper / Google STT / Azure STT — Chimege is the spec.
- ❌ Don't add a database. Files on disk only.
- ❌ Don't introduce React or any frontend framework. One HTML file.
- ❌ Don't try to identify *individual* students. All non-teachers = "Student".
- ❌ Don't average embeddings across all segments of a speaker. Use the longest segment.
- ❌ Don't skip the fallback. If matching fails, fall back gracefully with a flag.
- ❌ Don't write 12 abstractions before the pipeline works end-to-end. Happy path first.
- ❌ Don't commit `.env` or anything in `tmp/`.

---

## 13. Definition of Done (MVP)

- [ ] `pip install -r requirements.txt` succeeds in a clean venv
- [ ] `uvicorn app.main:app --reload` starts without error
- [ ] Visiting `http://localhost:8000` shows a page with **two** upload inputs
- [ ] Uploading a teacher reference + lecture video returns a valid `.xlsx`
- [ ] The xlsx has Teacher rows matching the reference voice and Student rows otherwise
- [ ] When matching fails (e.g., garbage reference), the fallback warning row appears
- [ ] README explains setup in <10 steps
- [ ] `.env.example` exists; `.env` is gitignored
