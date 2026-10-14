import json, os, tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from . import arena, brain, facility, facility_auth, free_models, image_client, lessons, models, store, video_client, video_understanding
from .orchestrator import orchestrate

app = FastAPI(title="AI Orchestra")
PAGE = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


class RunIn(BaseModel):
    prompt: str
    workspace_id: str = "default"


class ArenaIn(BaseModel):
    prompt: str
    model_ids: list


class WinnerIn(BaseModel):
    run_id: str
    winner_id: str


class WsIn(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    skills: Optional[list] = None


class SkillIn(BaseModel):
    name: str
    text: str


AGENTS = ["research", "writer", "coder", "factcheck", "planner", "synthesis", "direct"]


class FlagIn(BaseModel):
    id: str
    enabled: bool


class CustomIn(BaseModel):
    provider: str
    model: str
    label: str = ""


class SettingsIn(BaseModel):
    pins: Optional[dict] = None
    allow_private: Optional[bool] = None
    power: Optional[dict] = None


class KeyIn(BaseModel):
    provider: str
    key: str


class VideoIn(BaseModel):
    model_id: str
    prompt: str
    aspect_ratio: str = "16:9"


class ImageIn(BaseModel):
    provider: str
    model_id: str = ""
    prompt: str
    aspect_ratio: str = "1:1"


class FileIn(BaseModel):
    name: str
    text: str


class VideoAskIn(BaseModel):
    question: str


@app.post("/api/run")
async def run(body: RunIn):
    async def stream():
        async for ev in orchestrate(body.prompt, body.workspace_id):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/tasks/run")
async def run_facility_task(request: Request):
    """Facility task loop. Fails closed: needs FACILITY_API_TOKEN, and write tasks also need a signed, task-bound write grant."""
    try:
        facility_auth.check_api_token(request.headers.get("authorization"))
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body must be JSON")
        task = facility.validate_task(dict(body) if isinstance(body, dict) else body)
        if task["permissions"] == "write":
            facility_auth.check_write_grant(task, request.headers.get("x-facility-write-grant"))
        return await facility.run_guarded(task)
    except facility_auth.AuthError as e:
        raise HTTPException(e.status, e.message)
    except facility.TaskError as e:
        raise HTTPException(400, str(e))
    except facility.Busy as e:
        raise HTTPException(429, str(e))
    except facility.ExecutionDisabled as e:
        raise HTTPException(503, str(e))


@app.get("/api/workspaces")
def workspaces():
    return store.list_workspaces()


@app.post("/api/workspaces")
def new_workspace(body: WsIn):
    return {"id": store.create_workspace(body.name or "Untitled")}


@app.patch("/api/workspaces/{wid}")
def edit_workspace(wid: str, body: WsIn):
    store.update_workspace(wid, body.name, body.notes, body.skills)
    return {"ok": True}


@app.get("/api/skills")
def skills_list():
    return store.list_skills()


@app.post("/api/skills")
def skills_add(body: SkillIn):
    if not body.text.strip():
        raise HTTPException(400, "Paste some text for the skill")
    return {"id": store.add_skill(body.name, body.text)}


@app.delete("/api/skills/{sid}")
def skills_remove(sid: str):
    store.delete_skill(sid)
    return {"ok": True}


@app.get("/api/workspaces/{wid}/files")
def files(wid: str):
    return store.list_files(wid)


@app.post("/api/workspaces/{wid}/files")
def upload(wid: str, body: FileIn):
    return {"id": store.add_file(wid, body.name, body.text)}


@app.delete("/api/files/{fid}")
def remove_file(fid: str):
    store.delete_file(fid)
    return {"ok": True}


@app.get("/api/runs")
def runs(workspace_id: str = "default"):
    return store.list_runs(workspace_id)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@app.get("/api/models")
def get_models():
    return {"models": models.catalog(), "agents": AGENTS, "providers": list(free_models.PROVIDERS) + ["codex"],
            "pins": json.loads(store.get_setting("pins", "{}") or "{}"),
            "allow_private": store.get_setting("allow_private", "0") == "1"}


@app.get("/api/setup")
def setup_list():
    out = []
    for p, (url, note) in free_models.SIGNUP.items():
        is_chat = p in free_models.PROVIDERS
        out.append({"provider": p, "signup_url": url, "note": note,
                    "connected": free_models.has_key(p) if is_chat else bool(store.get_provider_key(p)),
                    "from_env": bool(os.getenv(free_models.key_var(p))) if is_chat else False})
    for p, (url, note) in image_client.PROVIDERS.items():
        if p == "gemini":
            continue  # reuses the Gemini key already listed above
        out.append({"provider": p, "signup_url": url, "note": note, "connected": image_client.has_key(p), "from_env": False})
    return out


@app.put("/api/setup/key")
def setup_set_key(body: KeyIn):
    if body.provider not in free_models.SIGNUP and body.provider not in image_client.PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    if os.getenv(free_models.key_var(body.provider)):
        raise HTTPException(400, "This key is set as an environment variable and can only be changed there")
    store.set_provider_key(body.provider, body.key)
    connected = free_models.has_key(body.provider) if body.provider in free_models.PROVIDERS else bool(body.key.strip())
    return {"ok": True, "connected": connected}


@app.post("/api/image")
async def image_generate(body: ImageIn):
    return await image_client.generate(body.provider, body.model_id, body.prompt, body.aspect_ratio)


@app.post("/api/video")
async def video_generate(body: VideoIn):
    async def stream():
        async for ev in video_client.generate(body.model_id, body.prompt, body.aspect_ratio):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/video/understand")
async def video_understand(file: UploadFile = File(...), workspace_id: str = Form("default")):
    """Upload a video and analyze it: audio extraction, timestamped transcription, optional frame
    analysis, structured result. Streams progress over SSE, same pattern as /api/run and /api/video."""
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_video.write(await file.read())
    tmp_video.close()

    async def stream():
        try:
            async for ev in video_understanding.analyze_video(tmp_video.name, "upload", None, file.filename):
                if ev["type"] == "final":
                    vid = os.urandom(4).hex()
                    store.save_video_analysis(vid, workspace_id, file.filename, "upload", None,
                                              ev["transcript"], ev["frame_notes"], ev["result"])
                    ev = {**ev, "id": vid}
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            try:
                os.unlink(tmp_video.name)
            except OSError:
                pass
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/video/understand")
def video_understand_list(workspace_id: str = "default"):
    return store.list_video_analyses(workspace_id)


@app.get("/api/video/understand/{vid}")
def video_understand_get(vid: str):
    r = store.get_video_analysis(vid)
    if not r:
        raise HTTPException(404, "Analysis not found")
    return r


@app.post("/api/video/understand/{vid}/ask")
async def video_understand_ask(vid: str, body: VideoAskIn):
    r = store.get_video_analysis(vid)
    if not r:
        raise HTTPException(404, "Analysis not found")
    answer = await video_understanding.ask_about_video(r["transcript"], r["result"], r["frame_notes"], body.question)
    return {"answer": answer}


@app.put("/api/models/enabled")
def set_enabled(body: FlagIn):
    store.set_flag(body.id, body.enabled)
    return {"ok": True}


@app.post("/api/models/custom")
def add_custom(body: CustomIn):
    if (body.provider not in free_models.PROVIDERS and body.provider != "codex") or not body.model.strip():
        raise HTTPException(400, "Pick a provider and enter a model id")
    return {"id": store.add_custom(body.provider, body.model.strip(), body.label)}


@app.delete("/api/models/custom")
def remove_custom(id: str):
    store.delete_custom(id)
    return {"ok": True}


@app.put("/api/settings")
def put_settings(body: SettingsIn):
    if body.pins is not None:
        store.set_setting("pins", json.dumps({k: v for k, v in body.pins.items() if v}))
    if body.power:
        cur = json.loads(store.get_setting("power", "{}") or "{}")
        cur.update({k: v for k, v in body.power.items() if isinstance(v, int) and 1 <= v <= 5})
        store.set_setting("power", json.dumps(cur))
    if body.allow_private is not None:
        store.set_setting("allow_private", "1" if body.allow_private else "0")
    return {"ok": True}


@app.post("/api/arena")
async def arena_run(body: ArenaIn):
    async def stream():
        async for ev in arena.run(body.prompt, body.model_ids):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/arena")
def arena_list():
    return store.list_arena()


@app.put("/api/arena/winner")
def arena_winner(body: WinnerIn):
    arena.set_winner(body.run_id, body.winner_id)
    return {"ok": True}


@app.get("/api/arena/export")
def arena_export():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(arena.export_jsonl(), media_type="application/jsonl",
                              headers={"Content-Disposition": "attachment; filename=training_pairs.jsonl"})


@app.post("/api/brain/tune")
def brain_tune():
    return {"changes": brain.tune()}


@app.get("/api/brain/log")
def brain_log():
    return store.tuning_log()


@app.get("/api/tally")
def get_tally():
    return store.tally_summary()


@app.delete("/api/tally")
def reset_tally():
    store.clear_tally()
    return {"ok": True}


class LessonIn(BaseModel):
    rule: str
    agent: str = "*"


class LessonStatusIn(BaseModel):
    status: str


@app.get("/api/lessons")
def get_lessons(status: Optional[str] = None):
    return lessons.list_lessons(status)


@app.post("/api/lessons")
def post_lesson(body: LessonIn):
    if not body.rule.strip():
        raise HTTPException(400, "rule is required")
    return {"id": lessons.add_lesson(body.rule, body.agent)}


@app.put("/api/lessons/{lid}")
def put_lesson(lid: str, body: LessonStatusIn):
    try:
        found = lessons.set_status(lid, body.status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not found:
        raise HTTPException(404, "lesson not found")
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(PAGE)
