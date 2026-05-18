import os
import time

import requests

CHIMEGE_BASE = os.getenv("CHIMEGE_BASE", "https://api.chimege.com/v1.2")
SUBMIT_PATH = os.getenv("CHIMEGE_SUBMIT_PATH", "/stt-long")
TRANSCRIPT_PATH = os.getenv("CHIMEGE_TRANSCRIPT_PATH", "/stt-long-transcript")
CONTENT_TYPE = os.getenv("CHIMEGE_CONTENT_TYPE", "application/octet-stream")

SUBMIT_TIMEOUT = 60
POLL_TIMEOUT = 60
POLL_MAX_SECONDS = 600
POLL_INTERVAL_SECONDS = 3.0
SUBMIT_RETRIES = 3

_DEBUG_LOGGED = {"submit": False, "poll": False, "done": False}


def transcribe(wav_bytes: bytes) -> str:
    """Submit audio to Chimege STT (async), then poll for the transcript.
    Returns '' if the request fails — don't kill the pipeline over one segment."""
    token = os.getenv("CHIMEGE_TOKEN")
    if not token:
        raise RuntimeError("CHIMEGE_TOKEN not set.")

    if len(wav_bytes) < 1024:
        print(f"[transcribe] skipping segment: payload {len(wav_bytes)}B is too small")
        return ""

    uuid = _submit(wav_bytes, token)
    if not uuid:
        print("[transcribe] empty result: submit returned no UUID")
        return ""
    text = _poll(uuid, token)
    if not text:
        print(f"[transcribe] empty result: no text returned for UUID {uuid}")
    return text


def _submit(wav_bytes: bytes, token: str) -> str:
    url = f"{CHIMEGE_BASE}{SUBMIT_PATH}"
    headers = {"Content-Type": CONTENT_TYPE, "Token": token}
    last_err: Exception | None = None

    for attempt in range(1, SUBMIT_RETRIES + 1):
        try:
            r = requests.post(url, data=wav_bytes, headers=headers, timeout=SUBMIT_TIMEOUT)
            if r.status_code == 200:
                if not _DEBUG_LOGGED["submit"]:
                    print(f"[transcribe] first submit response: {r.text[:300]}")
                    _DEBUG_LOGGED["submit"] = True
                uuid = _extract_uuid(r)
                if not uuid:
                    print(f"[transcribe] no UUID in submit response: {r.text[:200]}")
                return uuid
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                print(f"[transcribe] submit {r.status_code}: {r.text[:200]}")
                return ""
            last_err = RuntimeError(f"submit HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            last_err = e
        if attempt < SUBMIT_RETRIES:
            time.sleep(2.0 * attempt)

    print(f"[transcribe] submit failed after retries: {last_err}")
    return ""


def _poll(uuid: str, token: str) -> str:
    url = f"{CHIMEGE_BASE}{TRANSCRIPT_PATH}"
    headers = {"Token": token, "UUID": uuid}
    deadline = time.time() + POLL_MAX_SECONDS
    last_status = None

    while time.time() < deadline:
        try:
            r = requests.get(url, headers=headers, timeout=POLL_TIMEOUT)
            last_status = r.status_code
            if r.status_code == 200:
                if not _DEBUG_LOGGED["poll"]:
                    print(f"[transcribe] first transcript response: {r.text[:300]}")
                    _DEBUG_LOGGED["poll"] = True
                try:
                    data = r.json()
                except ValueError:
                    return r.text.strip()
                if isinstance(data, dict) and data.get("done") is False:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                if not _DEBUG_LOGGED["done"]:
                    print(f"[transcribe] first done response: {r.text[:500]}")
                    _DEBUG_LOGGED["done"] = True
                return _extract_text_from_data(data)
            if 400 <= r.status_code < 500 and r.status_code not in (404, 408, 425, 429):
                print(f"[transcribe] transcript {r.status_code} for {uuid}: {r.text[:200]}")
                return ""
        except requests.RequestException:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)

    print(f"[transcribe] polling timed out (last status {last_status}) for {uuid}")
    return ""


def _extract_uuid(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text.strip()
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("uuid", "UUID", "id", "request_id", "requestId"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        inner = data.get("data")
        if isinstance(inner, str):
            return inner.strip()
        if isinstance(inner, dict):
            for key in ("uuid", "UUID", "id"):
                value = inner.get(key)
                if isinstance(value, str) and value:
                    return value
    return ""


def _extract_text_from_data(data) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list):
       
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "word", "transcript"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return " ".join(p.strip() for p in parts if p).strip()
    if isinstance(data, dict):
        for key in ("transcription", "text", "transcript", "result", "transcribed_text"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                return _extract_text_from_data(value)
            if isinstance(value, dict):
                inner = _extract_text_from_data(value)
                if inner:
                    return inner
        inner = data.get("data")
        if inner is not None:
            return _extract_text_from_data(inner)
    return ""
