"""Does the orchestra beat one plain prompt? Usage: python eval.py [prompts.txt]
Each prompt runs both ways; a blind judge (order randomised) picks the better answer.
Writes eval_results.md. Uses many calls, so start with 5-10 prompts."""
import asyncio, random, sys
from app.claude_client import run_model
from app.orchestrator import orchestrate

JUDGE = "You compare two answers to the same request. Pick the better one for accuracy, completeness and clarity. Reply with exactly 'A' or 'B' on the first line, then a one-line reason."


async def orchestra(p):
    final = ""
    async for ev in orchestrate(p):
        if ev["type"] == "final":
            final = ev["text"]
        elif ev["type"] == "error":
            raise RuntimeError(ev["message"])
    return final


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_prompts.txt"
    prompts = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    score, rows = {"orchestra": 0, "single": 0, "tie": 0}, []
    for p in prompts:
        o, (s, _) = await asyncio.gather(orchestra(p), run_model("You are a helpful expert.", p, "main"))
        flip = random.random() < 0.5
        a, b = (s, o) if flip else (o, s)
        verdict, _ = await run_model(JUDGE, f"Request: {p}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}", "top")
        pick = verdict.strip()[:1].upper()
        if pick not in ("A", "B"):
            win = "tie"
        else:
            win = ("single" if pick == "A" else "orchestra") if flip else ("orchestra" if pick == "A" else "single")
        score[win] += 1
        rows.append(f"- **{win}** | {p} | {verdict.strip().splitlines()[-1][:120]}")
        print(win, "|", p)
    out = f"# Eval\n\nOrchestra {score['orchestra']} / Single {score['single']} / Tie {score['tie']}\n\n" + "\n".join(rows)
    open("eval_results.md", "w").write(out)
    print("\n" + out.splitlines()[2])


asyncio.run(main())
