"""One turn of a shared thread: the user's message goes to every active participant, each answer is saved, and a failing
participant is reported without stopping the others. Participants run read-only in an empty directory, so a chat turn
never touches a real workspace."""
import tempfile
from pathlib import Path

from . import facility, threads

# model each participant runs at for chat turns; chatgpt's empty model means the Codex CLI's own default
CHAT_MODELS = {"claude": "claude:sonnet", "chatgpt": ""}


async def run_turn(thread_id: str, content: str, timeout: float = 300, runners: dict = None) -> list:
    runners = runners or facility.RUNNERS
    content = content.strip()
    if not content:
        raise threads.ThreadError("content is required")
    # checked before anything is written: a disabled provider must not leave an unanswered user message in the thread
    facility.ensure_exec_enabled(False)
    participants = threads.active_participants(thread_id)
    if not participants:
        raise threads.ThreadError("no active participants: add a model to the thread first")
    threads.add_message(thread_id, "user", content)
    results = []
    for p in participants:
        # built per participant, so a later model also sees earlier answers from this same turn
        prompt = threads.prompt_for(thread_id, p)
        try:
            with tempfile.TemporaryDirectory(prefix="thread-") as ws:
                res = await runners[p](Path(ws), prompt, False, CHAT_MODELS[p], timeout)
            threads.add_message(thread_id, "assistant", res["text"], p)
            results.append({"participant": p, "ok": True, "text": res["text"]})
        except Exception as e:  # one model failing must not stop the others
            results.append({"participant": p, "ok": False, "error": str(e)[:300]})
    return results
