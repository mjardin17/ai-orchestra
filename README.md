# AI Orchestra

Setup (personal use, runs on your own Claude subscription)
1. Install Claude Code, then run `claude login`.
2. `cd backend && pip install -r requirements.txt`
3. `uvicorn app.main:app --reload`, then open http://localhost:8000

What it does
- Router: answers simple requests directly; otherwise plans up to 3 stages of specialists, picks a model tier per agent, and the fact-checker can trigger one revision pass.
- Workspaces: each has notes and uploaded text files that every agent reads, plus the last 3 exchanges for follow-ups.
- Research agent uses web search (retries without it if your setup rejects it).
- Stop button cancels the run and its background calls. Export markdown, or Print / PDF from the browser.
- `python eval.py` compares the orchestra with a single plain prompt using a blind judge. Edit `eval_prompts.txt` first.

Models tab (dashboard)
- ChatGPT without an API key: install OpenAI's Codex CLI (`npm i -g @openai/codex`), run `codex login` and sign in with your ChatGPT account, restart, then tick "ChatGPT (your login)". It uses your plan's Codex allowance. To pick a specific ChatGPT model, use Add a model with provider `codex` and the model id.
- Tick which models are in your pool: Claude Haiku/Sonnet/Opus, plus free-tier models from Gemini, Groq and Mistral. DeepSeek (V4 Flash and Pro) is also listed: it needs a `DEEPSEEK_API_KEY` and bills per token once any trial credits run out. Add any other model id (e.g. an OpenRouter or SambaNova free model) with the Add form.
- Free models need a free key from the provider's site. Copy `backend/.env.example`, fill in what you want, restart, then tick them.
- The council brain (a Claude router) rates each job easy/medium/hard, then starts on the cheapest enabled model whose capability fits and escalates only if the model errors or a quick quality check says the result isn't good enough. Capability (1-5) and cost are rough defaults: adjust capability per model in the tab. Pin a model to an agent to skip all of this for that agent.
- A free model that errors or is rate-limited falls back to Claude Haiku. Free tiers may train on your prompts, so workspaces with notes or files use Claude only unless you allow it.

Settings (environment variables)
- `MAX_TIER` (default `top`): hard ceiling. `MAX_TIER=main` keeps Opus off to save plan limits.
- `MODEL_FAST`, `MODEL_MAIN`, `MODEL_TOP`: override the Claude model names.
- `ORCHESTRA_PROVIDER=api` plus `ANTHROPIC_API_KEY`: use an API key (required to host this for other people; subscription logins are personal-use only).
- Lower `LIMIT` in orchestrator.py if you hit plan limits. Check `claude --help` if your CLI rejects `-p`, `--model` or `--allowedTools`.
