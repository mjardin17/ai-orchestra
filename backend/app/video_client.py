"""Higgsfield video generation: submit a job, poll until done. Paid service (register a card and
top up a balance in their dashboard; new accounts sometimes get starter credit - check their site).
Docs change model ids over time: paste the exact model_id from https://docs.higgsfield.ai/docs/models
into the app rather than assuming one here."""
import asyncio, json, urllib.error, urllib.request
from . import store

BASE = "https://api.higgsfield.ai"
DOCS = "https://docs.higgsfield.ai/docs/models/video-generation"


def creds():
    raw = store.get_provider_key("higgsfield") or ""
    if ":" not in raw:
        return None
    key_id, secret = raw.split(":", 1)
    return key_id, secret


def _req(method, url, key_id, secret, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {
        "Authorization": f"Key {key_id}:{secret}", "Content-Type": "application/json", "Accept": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


async def generate(model_id: str, prompt: str, aspect_ratio="16:9", max_wait=240):
    """Yields status dicts as the job progresses, ending with one that has 'video' or 'error'."""
    c = creds()
    if not c:
        yield {"error": "No Higgsfield key saved. Add KEY_ID:KEY_SECRET in Setup first."}
        return
    key_id, secret = c
    try:
        sub = await asyncio.to_thread(_req, "POST", f"{BASE}/{model_id}", key_id, secret,
                                      {"prompt": prompt, "aspect_ratio": aspect_ratio})
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        yield {"error": f"HTTP {e.code}: {detail or 'request rejected'}. Check the model_id against {DOCS}"}
        return
    except Exception as e:
        yield {"error": str(e)[:300]}
        return

    status_url = sub.get("status_url")
    yield {"status": sub.get("status", "queued"), "request_id": sub.get("request_id")}
    waited = 0
    while status_url and waited < max_wait:
        await asyncio.sleep(5)
        waited += 5
        try:
            s = await asyncio.to_thread(_req, "GET", status_url, key_id, secret)
        except Exception as e:
            yield {"error": str(e)[:300]}
            return
        st = s.get("status")
        if st == "completed":
            video = (s.get("video") or {}).get("url") or (s.get("videos") or [{}])[0].get("url")
            yield {"status": "completed", "video": video}
            return
        if st in ("failed", "nsfw"):
            yield {"status": st, "error": s.get("error") or f"Job ended with status '{st}'"}
            return
        yield {"status": st or "processing"}
    if status_url:
        yield {"error": "Timed out waiting for the video. It may still finish - check Higgsfield's dashboard."}
