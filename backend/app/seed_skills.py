"""Joshua's own skill library: rules learned from his real BossLister / Penny Deals / Empire OS work, so the
skill selector can hand each agent the right ones. Re-running is safe: skills are matched by name, only
changed text is rewritten, nothing is duplicated.

Run once from the backend folder:  python -m app.seed_skills
"""
import time, uuid
from . import store

SKILLS = [
    {"name": "Josh: working style",
     "text": "Start every reply to Josh with 'Josh'. Short, plain answers, no jargon. Verify before claiming: run the "
             "check, then say what you saw. Never redo work already done or re-audit a project he says is finished. "
             "Ask only when truly blocked; otherwise pick the sensible default and say so. Never ask for tokens, "
             "passwords or keys in chat. Commit locally only; never push, delete, or publish without his clear yes."},
    {"name": "Josh: listing titles",
     "text": "Write marketplace listing titles (eBay, Shopify, Etsy) of 80 characters or fewer: brand, model, key "
             "feature, size or count, condition. No ALL CAPS shouting, no filler words, no emojis. Never invent "
             "specs, condition or authenticity; if the photo or data doesn't show it, leave it out. Use full-size "
             "photo URLs (eBay s-l1600, not the s-l140 thumbnail). Check the title doesn't repeat brand or category."},
    {"name": "Josh: Instagram post",
     "text": "Instagram product post for @jardinsoutpost (a Business account): one clear photo, caption = item title, "
             "price, one benefit line, 'link in bio' (links in captions aren't clickable), 5-8 relevant hashtags. "
             "Always show a dry-run preview of the exact photo and caption first; a post is public, so publish "
             "only after Josh says yes to that exact item."},
    {"name": "Josh: token fixes",
     "text": "Diagnose logins by asking the platform live, never by assumption. Facebook: Graph API Explorer tokens "
             "are short-lived; exchange for a long-lived Page token so it stops expiring. eBay: invalid_grant means "
             "the refresh token was revoked or replaced; a scope error means the token lacks sell.inventory or "
             "sell.account, so reconnect with the Connect button. Instagram: renew the long-lived token before "
             "60 days. Tokens are saved in .env.local (back it up first); never print or request them in chat. "
             "Run 'npm run token-health' for a read-only check."},
    {"name": "Josh: dry run before live",
     "text": "Anything public or irreversible (publishing a listing, posting, deleting, pushing code, changing "
             "repo visibility) gets a dry run first showing exactly what will happen, then waits for Josh's clear "
             "yes for that specific action. One approval doesn't carry to the next action. Report what was "
             "verified and what was not run."},
    {"name": "Josh: cross-list oversell guard",
     "text": "Most items are quantity 1 and already live on eBay. Before listing the same item on a second channel "
             "(Shopify, Facebook, Instagram), warn about overselling, import as Draft, and publish only items Josh "
             "approves. Sales must end the other listings: confirm the inventory sync is on before publishing."},
    {"name": "Josh: card pricing",
     "text": "Price trading cards and collectibles from real sold comps (completed sales), not asking prices. State "
             "the comp count, price range and median. Adjust for condition or grade, subtract fees per platform "
             "(eBay final value fee, payment fee, shipping), and show net profit. If comps are too few or the "
             "match is unsure, say so instead of guessing a number."},
    {"name": "Josh: Penny Deals verdict",
     "text": "Penny Deals is a separate project from BossLister; never mix their code. Verdict rule for an in-store "
             "scan (buy or don't buy): flag a real spread only when live comps are confirmed, the title match is "
             "confident, every fee comes from a verified table, and profit clears $10 and 50% ROI. Otherwise say "
             "'don't buy' or 'not enough data'; never invent numbers."},
    {"name": "Josh: Shopify import",
     "text": "Shopify product imports go in as Draft (Status=draft, Published=FALSE) so nothing is public until "
             "Josh publishes it. Use the Jardins Outpost store, SKU from the database, quantity tracked by "
             "Shopify, full-size photos. Generated files go in the exports folder, which stays out of git."},
    {"name": "Josh: PC care",
     "text": "Josh's PC has very little free disk (under 1 GB on drive C) and little RAM. Before installing "
             "dependencies, large downloads or graphs, check free space. Never delete his project folders or "
             "agent apps (OpenClaw, Hermes). Cleaning caches (Temp, pnpm store, Playwright browsers) is fine only "
             "after he approves. Save notes to memory early, because a full disk makes writes fail."},
]


def seed():
    """Adds or refreshes every skill above; returns how many rows were written (0 when already current)."""
    written = 0
    for sk in SKILLS:
        row = store.DB.execute("select id, text from skills where name=?", (sk["name"],)).fetchone()
        if row is None:
            store.DB.execute("insert into skills values(?,?,?,?)", (uuid.uuid4().hex[:8], sk["name"], sk["text"], time.time()))
            written += 1
        elif row["text"] != sk["text"]:
            store.DB.execute("update skills set text=? where id=?", (sk["text"], row["id"]))
            written += 1
    store.DB.commit()
    return written


if __name__ == "__main__":
    print(f"Seeded {seed()} skill(s); library now has {len(store.list_skills())}.")
