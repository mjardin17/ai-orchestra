"""Video understanding: audio extraction, timestamped transcription, optional visual sampling,
and a structured analysis - summary, claims, objects, events. Kept decoupled from acquisition:
analyze_video() takes a local video_path and a source_type/source_ref pair. Today source_type is
always "upload" (from the Understand tab); a later Composio-based Facebook/Instagram/Whatnot
connector would just download a video to a temp path and call this same function with
source_type="facebook" (etc.) and source_ref=the post/video URL or ID - no change needed here.

Requires ffmpeg/ffprobe on PATH for audio+frame extraction (no extra Python dependency).
Transcription uses Groq's Whisper endpoint (reuses the same GROQ_API_KEY already used for chat).
Visual frame description uses Gemini (reuses the same GEMINI_API_KEY already used for chat/image)
and is skipped - not an error - if no Gemini key is configured, matching "analyze visually when
possible" rather than treating vision as required.
"""
import asyncio, base64, json, mimetypes, os, re, shutil, subprocess, tempfile, time, urllib.error, urllib.request, uuid
from . import free_models
from .claude_client import run_agent
from .prompts import VIDEO_ANALYSIS, VIDEO_ASK

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TRANSCRIBE_MODEL = os.getenv("GROQ_TRANSCRIBE_MODEL", "whisper-large-v3")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest")
FRAME_INTERVAL_SEC = 8
MAX_FRAMES = 12


class Unsupported(RuntimeError):
    """Raised for a clear, user-facing setup problem (missing ffmpeg, missing key) - not escalated/retried."""


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b"").decode(errors="replace")[-500:] or f"command failed: {' '.join(cmd)}")
    return r.stdout


def extract_audio(video_path: str, out_path: str):
    """16kHz mono WAV - the format Whisper-style transcription expects."""
    if not shutil.which("ffmpeg"):
        raise Unsupported("ffmpeg is not installed/not on PATH. Install ffmpeg to use Video Understanding.")
    _run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_path])


def probe_duration(video_path: str) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", video_path])
        return float(out.decode().strip())
    except Exception:
        return 0.0


def extract_keyframes(video_path: str, out_dir: str, interval_sec=FRAME_INTERVAL_SEC, max_frames=MAX_FRAMES):
    """Samples one frame every interval_sec, capped at max_frames. Returns [(timestamp_seconds, path), ...]."""
    if not shutil.which("ffmpeg"):
        return []
    duration = probe_duration(video_path)
    if duration and duration / interval_sec > max_frames:
        interval_sec = max(1, duration / max_frames)
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    try:
        _run(["ffmpeg", "-y", "-i", video_path, "-vf", f"fps=1/{interval_sec}", "-frames:v", str(max_frames), pattern])
    except Exception:
        return []  # visual analysis is best-effort; a frame-extraction failure should not kill the whole run
    frames = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_"))
    return [(round(i * interval_sec, 1), os.path.join(out_dir, f)) for i, f in enumerate(frames)]


def _multipart_post(url, headers, fields: dict, file_field: str, file_path: str):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{os.path.basename(file_path)}"\r\n'
        f'Content-Type: {mime}\r\n\r\n'.encode() + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, body, {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


async def transcribe_groq(audio_path: str):
    """Returns [{"start": float, "end": float, "text": str}, ...]. Raises Unsupported if no Groq key is set."""
    key = free_models.get_key("groq")
    if not key:
        raise Unsupported("No Groq key saved. Transcription needs GROQ_API_KEY - add it in Models \u2192 Setup.")
    try:
        data = await asyncio.to_thread(_multipart_post, GROQ_TRANSCRIBE_URL, {"Authorization": f"Bearer {key}"},
            {"model": GROQ_TRANSCRIBE_MODEL, "response_format": "verbose_json"}, "file", audio_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Groq transcription HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    segments = data.get("segments") or []
    if segments:
        return [{"start": round(s.get("start", 0), 1), "end": round(s.get("end", 0), 1), "text": s.get("text", "").strip()}
                for s in segments if s.get("text", "").strip()]
    text = (data.get("text") or "").strip()  # provider returned plain text with no segment timing
    return [{"start": 0.0, "end": 0.0, "text": text}] if text else []


async def describe_frame_gemini(image_path: str):
    """Returns a short description string, or None if Gemini isn't configured or the call fails
    (visual analysis is optional - a failure here should not fail the whole analysis)."""
    key = free_models.get_key("gemini")
    if not key:
        return None
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent?key={key}"
        body = {"contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
            {"text": "In one short sentence, describe what is visibly happening in this video frame: setting, people, objects, action."}]}]}
        data = await asyncio.to_thread(
            lambda: json.load(urllib.request.urlopen(
                urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60)))
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = next((p["text"] for p in parts if "text" in p), None)
        return text.strip() if text else None
    except Exception:
        return None


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


async def analyze_with_claude(transcript, frame_notes, source_meta):
    transcript_block = "\n".join(f"[{_fmt_ts(t['start'])}] {t['text']}" for t in transcript) or "(no speech detected)"
    frames_block = "\n".join(f"[{_fmt_ts(ts)}] {desc}" for ts, desc in frame_notes) if frame_notes else "(no frame analysis available)"
    prompt = (f"Source: {source_meta.get('source_type', 'upload')} - {source_meta.get('filename', 'uploaded video')}\n\n"
              f"Timestamped transcript:\n{transcript_block}\n\nFrame notes:\n{frames_block}")
    raw = await run_agent(VIDEO_ANALYSIS, prompt, "main")
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("no JSON object in response")
        data = json.loads(m.group(0))
        return {
            "summary": data.get("summary") or "",
            "claims": [c for c in data.get("claims", []) if isinstance(c, str)][:20],
            "objects": [o for o in data.get("objects", []) if isinstance(o, str)][:20],
            "events": [{"timestamp": float(e.get("timestamp", 0)), "label": str(e.get("label", ""))}
                       for e in data.get("events", []) if isinstance(e, dict)][:30],
        }
    except Exception:
        return {"summary": raw.strip()[:800] or "Analysis did not return structured output.", "claims": [], "objects": [], "events": []}


async def analyze_video(video_path: str, source_type="upload", source_ref=None, filename=None, want_visual=True):
    """Async generator: yields progress events, then one {"type": "final", "transcript":..., "result":...}.
    Mirrors the same yield-progress-then-final pattern used by orchestrator.orchestrate / arena.run / video_client.generate."""
    tmp = tempfile.mkdtemp(prefix="vu_")
    source_meta = {"source_type": source_type, "source_ref": source_ref, "filename": filename or os.path.basename(video_path)}
    try:
        yield {"type": "extracting_audio"}
        audio_path = os.path.join(tmp, "audio.wav")
        try:
            extract_audio(video_path, audio_path)
        except Unsupported as e:
            yield {"type": "error", "message": str(e)}
            return

        yield {"type": "transcribing"}
        try:
            transcript = await transcribe_groq(audio_path)
        except Unsupported as e:
            yield {"type": "error", "message": str(e)}
            return

        frame_notes = []
        if want_visual and free_models.get_key("gemini"):
            yield {"type": "analyzing_frames"}
            frame_dir = os.path.join(tmp, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            frames = extract_keyframes(video_path, frame_dir)
            for ts, path in frames:
                desc = await describe_frame_gemini(path)
                if desc:
                    frame_notes.append((ts, desc))

        yield {"type": "synthesizing"}
        result = await analyze_with_claude(transcript, frame_notes, source_meta)
        yield {"type": "final", "transcript": transcript, "frame_notes": [{"timestamp": t, "description": d} for t, d in frame_notes],
               "result": result, "source": source_meta}
    except Exception as e:
        yield {"type": "error", "message": str(e)[:400]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def ask_about_video(transcript, result, frame_notes, question):
    transcript_block = "\n".join(f"[{_fmt_ts(t['start'])}] {t['text']}" for t in transcript) or "(no speech detected)"
    frames_block = "\n".join(f"[{_fmt_ts(f['timestamp'])}] {f['description']}" for f in frame_notes) if frame_notes else "(no frame analysis)"
    context = (f"Transcript:\n{transcript_block}\n\nFrame notes:\n{frames_block}\n\n"
               f"Prior analysis summary: {result.get('summary', '')}\n\nQuestion: {question}")
    return await run_agent(VIDEO_ASK, context, "main")
