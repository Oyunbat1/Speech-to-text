import os
from pathlib import Path
from typing import Callable

from app.audio import extract_audio, slice_audio
from app.diarize import diarize
from app.embed import assign_roles
from app.sheet import write_xlsx
from app.transcribe import transcribe


def process_lecture(
    lecture_path: str,
    teacher_reference_path: str,
    output_xlsx_path: str,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    """End-to-end pipeline. See CLAUDE.md section 6."""

    def report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        print(f"[pipeline] {msg}")

    threshold = float(os.getenv("TEACHER_MATCH_THRESHOLD", "0.5"))

    lecture_path = str(lecture_path)
    teacher_reference_path = str(teacher_reference_path)
    output_xlsx_path = str(output_xlsx_path)

    work_dir = Path(output_xlsx_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(lecture_path).stem
    lecture_wav = str(work_dir / f"{stem}__lecture.wav")
    teacher_wav = str(work_dir / f"{stem}__teacher.wav")

    try:
        report("Extracting audio from lecture...")
        extract_audio(lecture_path, lecture_wav)

        report("Extracting audio from teacher reference...")
        extract_audio(teacher_reference_path, teacher_wav)

        report("Diarizing lecture (this can take a while)...")
        segments = diarize(lecture_wav)
        report(f"Found {len(segments)} segments across speakers.")

        if not segments:
            write_xlsx([], [], output_xlsx_path, fallback_warning=False)
            return

        report("Matching speakers against teacher reference...")
        segments, info = assign_roles(
            segments,
            lecture_wav=lecture_wav,
            teacher_reference_wav=teacher_wav,
            threshold=threshold,
        )
        report(
            f"Best similarity = {info['best_similarity']:.3f}, "
            f"fallback used = {info['fallback_used']}"
        )

        report(f"Transcribing {len(segments)} segments via Chimege...")
        texts: list[str] = []
        for i, seg in enumerate(segments, start=1):
            if i % 10 == 0 or i == len(segments):
                report(f"  transcribing {i}/{len(segments)}")
            wav_bytes = slice_audio(lecture_wav, seg.start, seg.end)
            texts.append(transcribe(wav_bytes))

        report("Writing xlsx...")
        write_xlsx(
            segments,
            texts,
            output_xlsx_path,
            fallback_warning=info["fallback_used"],
        )
        report("Done.")
    finally:
        for path in (lecture_wav, teacher_wav):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
