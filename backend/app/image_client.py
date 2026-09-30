"""Photo/image generation. Unlike video, these are single-call APIs (no job polling).
Endpoint shapes below match each provider's docs as of writing - if a provider changes
its API, the error message returned will show what came back so you can adjust the model_id
or check https://ai.google.dev/gemini-api/docs/image-generation (Gemini), https://platform.stability.ai/docs
(Stability) or https://developer.ideogram.ai (Ideogram)."""
import base64, json, urllib.error, urllib.request
from . import free_models, store

PROVIDERS = {  # name: signup url, note
    "gemini": ("https://aistudio.google.com/apikey", "Uses the same Google key as Gemini chat, if you already added one."),
    "stability": ("https://platform.stability.ai/account/keys", "Stability AI. Free trial credit on new accounts, then pay-per-image."),
    "ideogram": ("https://ideogram.ai/manage-api", "Ideogram. Best at readable text inside images. Free tier is limited/slow, then pay-per-image."),
}


def get_key(p):
    return free_models.get_key("gemini") if p == "gemini" else store.get_provider_key(p)


def has_key(p):
    return bool(get_key(p))


def _post_json(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


async def generate(provider: str, model_id: str, prompt: str, aspect_ratio="1:1"):
    """Returns {'image': 'data:image/png;base64,...' or a URL} or {'error': ...}."""
    import asyncio
    key = get_key(provider)
    if not key:
        return {"error": f"No key saved for {provider}. Add it in Setup first."}
    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={key}"
            data = await asyncio.to_thread(_post_json, url, {},
                {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseModalities": ["IMAGE"]}})
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            img = next((p["inlineData"] for p in parts if "inlineData" in p), None)
            if not img:
                return {"error": "No image in Gemini's response: " + json.dumps(data)[:300]}
            return {"image": f"data:{img['mimeType']};base64,{img['data']}"}

        if provider == "stability":
            url = "https://api.stability.ai/v2beta/stable-image/generate/" + (model_id or "core")
            data = await asyncio.to_thread(_post_json, url, {"Authorization": f"Bearer {key}", "Accept": "application/json"},
                {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png"})
            if "image" not in data:
                return {"error": "No image in Stability's response: " + json.dumps(data)[:300]}
            return {"image": f"data:image/png;base64,{data['image']}"}

        if provider == "ideogram":
            url = "https://api.ideogram.ai/generate"
            data = await asyncio.to_thread(_post_json, url, {"Api-Key": key},
                {"image_request": {"prompt": prompt, "aspect_ratio": aspect_ratio.replace(":", "x"), "model": model_id or "V_2"}})
            item = (data.get("data") or [{}])[0]
            if not item.get("url"):
                return {"error": "No image in Ideogram's response: " + json.dumps(data)[:300]}
            return {"image": item["url"]}

        return {"error": f"Unknown image provider: {provider}"}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"error": str(e)[:300]}
