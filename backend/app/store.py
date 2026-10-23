"""SQLite storage: workspaces (with notes), uploaded text files, saved runs, and context building."""
import json, sqlite3, time, uuid

DB = sqlite3.connect("orchestra.db", check_same_thread=False)
DB.row_factory = sqlite3.Row
DB.executescript("""
create table if not exists workspaces(id text primary key, name text, notes text default '');
create table if not exists files(id text primary key, workspace_id text, name text, text text);
create table if not exists runs(id text primary key, prompt text, route text, outputs text, final text, created real);
create table if not exists settings(key text primary key, value text);
create table if not exists model_flags(id text primary key, enabled int);
create table if not exists custom_models(id text primary key, provider text, model text, label text);
create table if not exists tally(model_id text, label text, escalated int, run_id text);
create table if not exists arena_runs(id text primary key, prompt text, results text, winner text, created real);
create table if not exists tuning_log(created real, changes text);
create table if not exists skills(id text primary key, name text, text text, created real);
create table if not exists provider_keys(provider text primary key, key text);
create table if not exists video_analyses(id text primary key, workspace_id text, filename text,
    source_type text, source_ref text, transcript text, frame_notes text, result text, created real);
insert or ignore into workspaces(id, name, notes) values('default','My workspace','');
""")
try:
    DB.execute("alter table runs add column workspace_id text default 'default'")
except sqlite3.OperationalError:
    pass
try:
    DB.execute("alter table workspaces add column skills text default '[]'")
except sqlite3.OperationalError:
    pass


def list_workspaces():
    return [dict(r) for r in DB.execute("select id, name, notes, skills from workspaces order by rowid")]


def create_workspace(name):
    wid = uuid.uuid4().hex[:8]
    DB.execute("insert into workspaces(id, name) values(?,?)", (wid, name.strip() or "Untitled"))
    DB.commit()
    return wid


def update_workspace(wid, name=None, notes=None, skills=None):
    if name is not None:
        DB.execute("update workspaces set name=? where id=?", (name.strip() or "Untitled", wid))
    if notes is not None:
        DB.execute("update workspaces set notes=? where id=?", (notes, wid))
    if skills is not None:
        DB.execute("update workspaces set skills=? where id=?", (json.dumps(skills), wid))
    DB.commit()


def add_file(wid, name, text):
    fid = uuid.uuid4().hex[:8]
    DB.execute("insert into files values(?,?,?,?)", (fid, wid, name, text[:200_000]))
    DB.commit()
    return fid


def list_files(wid, with_text=False):
    cols = "id, name, text" if with_text else "id, name"
    return [dict(r) for r in DB.execute(f"select {cols} from files where workspace_id=? order by rowid", (wid,))]


def delete_file(fid):
    DB.execute("delete from files where id=?", (fid,))
    DB.commit()


def has_private(wid):
    ws = DB.execute("select notes from workspaces where id=?", (wid,)).fetchone() if wid else None
    return bool((ws and ws["notes"].strip()) or (wid and list_files(wid)))


def add_skill(name, text):
    sid = uuid.uuid4().hex[:8]
    DB.execute("insert into skills values(?,?,?,?)", (sid, name.strip() or "Untitled skill", text[:200_000], time.time()))
    DB.commit()
    return sid


def list_skills():
    return [dict(r) for r in DB.execute("select id, name from skills order by rowid")]


def delete_skill(sid):
    DB.execute("delete from skills where id=?", (sid,))
    DB.commit()


def context(wid, limit=12000):
    """Skills + notes + uploaded files + last 3 exchanges, so follow-ups and every agent share the same background."""
    if not wid:
        return ""
    ws = DB.execute("select notes, skills from workspaces where id=?", (wid,)).fetchone()
    parts = []
    for sid in json.loads(ws["skills"] or "[]") if ws else []:
        sk = DB.execute("select name, text from skills where id=?", (sid,)).fetchone()
        if sk:
            parts.append(f"Skill \"{sk['name']}\" (follow this style/knowledge):\n{sk['text']}")
    if ws and ws["notes"].strip():
        parts.append("Workspace notes:\n" + ws["notes"])
    parts += [f"File {f['name']}:\n{f['text']}" for f in list_files(wid, True)]
    out = ["\n\n".join(parts)[:limit]] if parts else []
    recent = DB.execute("select prompt, final from runs where workspace_id=? order by created desc limit 3", (wid,)).fetchall()
    if recent:
        out.append("Earlier in this conversation:\n" + "\n\n".join(
            f"User: {r['prompt']}\nAnswer: {r['final'][:1500]}" for r in reversed(recent)))
    return "\n\n".join(out)


def save_run(rid, wid, prompt, route, outputs, final):
    DB.execute("insert into runs(id, prompt, route, outputs, final, created, workspace_id) values(?,?,?,?,?,?,?)",
               (rid, prompt, json.dumps(route), json.dumps(outputs), final, time.time(), wid))
    DB.commit()


def list_runs(wid="default"):
    rows = DB.execute("select id, prompt from runs where workspace_id=? order by created desc limit 50", (wid,))
    return [dict(r) for r in rows]


def get_run(rid):
    r = DB.execute("select * from runs where id=?", (rid,)).fetchone()
    return r and {"id": r["id"], "prompt": r["prompt"], "route": json.loads(r["route"]),
                  "outputs": json.loads(r["outputs"]), "final": r["final"]}


def get_setting(key, default=""):
    r = DB.execute("select value from settings where key=?", (key,)).fetchone()
    return r[0] if r else default


def set_setting(key, value):
    DB.execute("insert or replace into settings values(?,?)", (key, value))
    DB.commit()


def model_flags():
    return {r[0]: bool(r[1]) for r in DB.execute("select id, enabled from model_flags")}


def set_flag(mid, enabled):
    DB.execute("insert or replace into model_flags values(?,?)", (mid, int(enabled)))
    DB.commit()


def list_custom():
    return [dict(r) for r in DB.execute("select id, provider, model, label from custom_models order by rowid")]


def add_custom(provider, model, label):
    mid = f"{provider}:{model}"
    DB.execute("insert or replace into custom_models values(?,?,?,?)", (mid, provider, model, label.strip() or model))
    DB.commit()
    return mid


def delete_custom(mid):
    DB.execute("delete from custom_models where id=?", (mid,))
    DB.execute("delete from model_flags where id=?", (mid,))
    DB.commit()


def record_use(model_id, label, escalated, run_id, power=3):  # power kept only for brain.tune()'s "default_power"
    DB.execute("insert into tally values(?,?,?,?)", (model_id, label, int(escalated), run_id))
    DB.commit()


def tally_summary(limit_runs=200):
    rows = DB.execute("""
        select model_id, label, count(*) jobs, sum(escalated) esc from tally
        where run_id in (select distinct run_id from tally order by rowid desc limit ?) and model_id is not null
        group by model_id order by jobs desc
    """, (limit_runs,)).fetchall()
    return [{"id": r["model_id"], "label": r["label"], "jobs": r["jobs"], "escalated": r["esc"] or 0} for r in rows]


def clear_tally():
    DB.execute("delete from tally")
    DB.commit()


def save_arena(rid, prompt, results, suggested):
    DB.execute("insert into arena_runs(id, prompt, results, winner, created) values(?,?,?,?,?)",
               (rid, prompt, json.dumps(results), suggested, time.time()))
    DB.commit()


def set_arena_winner(rid, winner_id):
    DB.execute("update arena_runs set winner=? where id=?", (winner_id, rid))
    DB.commit()


def list_arena(full=False, limit=50):
    rows = DB.execute("select id, prompt, results, winner from arena_runs order by created desc limit ?", (limit,)).fetchall()
    out = [{"id": r["id"], "prompt": r["prompt"], "results": json.loads(r["results"]), "winner": r["winner"]} for r in rows]
    return out if full else [{"id": o["id"], "prompt": o["prompt"], "models": [x["label"] for x in o["results"]], "winner": o["winner"]} for o in out]


def log_tuning(changes):
    DB.execute("insert into tuning_log values(?,?)", (time.time(), json.dumps(changes)))
    DB.commit()


def tuning_log(limit=20):
    rows = DB.execute("select created, changes from tuning_log order by created desc limit ?", (limit,)).fetchall()
    return [{"created": r["created"], "changes": json.loads(r["changes"])} for r in rows]


def set_provider_key(provider, key):
    key = key.strip()
    if key:
        DB.execute("insert or replace into provider_keys values(?,?)", (provider, key))
    else:
        DB.execute("delete from provider_keys where provider=?", (provider,))
    DB.commit()


def get_provider_key(provider):
    r = DB.execute("select key from provider_keys where provider=?", (provider,)).fetchone()
    return r[0] if r else None


def stored_providers():
    return [r[0] for r in DB.execute("select provider from provider_keys")]


def save_video_analysis(vid, wid, filename, source_type, source_ref, transcript, frame_notes, result):
    DB.execute("insert into video_analyses values(?,?,?,?,?,?,?,?,?)",
               (vid, wid, filename, source_type, source_ref, json.dumps(transcript), json.dumps(frame_notes),
                json.dumps(result), time.time()))
    DB.commit()


def get_video_analysis(vid):
    r = DB.execute("select * from video_analyses where id=?", (vid,)).fetchone()
    return r and {"id": r["id"], "workspace_id": r["workspace_id"], "filename": r["filename"],
                  "source_type": r["source_type"], "source_ref": r["source_ref"],
                  "transcript": json.loads(r["transcript"]), "frame_notes": json.loads(r["frame_notes"]),
                  "result": json.loads(r["result"])}


def list_video_analyses(wid="default"):
    rows = DB.execute("select id, filename, created from video_analyses where workspace_id=? order by created desc limit 50", (wid,))
    return [dict(r) for r in rows]
