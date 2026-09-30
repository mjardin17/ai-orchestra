"""The self-tuning part of the 'brain': adjusts each model's capability rating from the escalation
tally, so the council's cheapest-capable-first ordering improves with use. This tunes ROUTING, not a
new model - it never changes what any model can actually do."""
import json
from . import store

MIN_JOBS = 6


def tune():
    power = json.loads(store.get_setting("power", "{}") or "{}")
    changes = []
    for row in store.tally_summary(limit_runs=500):
        mid = row.get("id")
        if not mid or row["jobs"] < MIN_JOBS:
            continue
        rate = row["escalated"] / row["jobs"]
        cur = power.get(mid, row.get("default_power", 3))
        new = cur
        if rate > 0.5 and cur > 1:      # frequently needed a stronger model: it's overrated, lower it
            new = cur - 1
        elif rate == 0 and row["jobs"] >= MIN_JOBS * 2 and cur < 5:  # never needed help: it can likely take harder jobs
            new = cur + 1
        if new != cur:
            power[mid] = new
            changes.append({"id": mid, "label": row["label"], "from": cur, "to": new, "jobs": row["jobs"], "rate": round(rate, 2)})
    if changes:
        store.set_setting("power", json.dumps(power))
        store.log_tuning(changes)
    return changes
