from app import seed_skills, skill_selector as ss, store


def setup_function():
    store.DB.execute("delete from skills"); store.DB.commit()


def test_seed_is_idempotent():
    first = seed_skills.seed()
    second = seed_skills.seed()
    assert first > 0 and second == 0
    assert store.DB.execute("select count(*) from skills").fetchone()[0] == len(seed_skills.SKILLS)


def test_seed_updates_changed_text_without_duplicating():
    seed_skills.seed()
    store.DB.execute("update skills set text='old' where name=?", (seed_skills.SKILLS[0]["name"],)); store.DB.commit()
    assert seed_skills.seed() == 1
    assert store.DB.execute("select count(*) from skills").fetchone()[0] == len(seed_skills.SKILLS)


def test_each_job_type_selects_its_own_skill():
    seed_skills.seed()
    jobs = {
        "write an ebay listing title for this item": "Josh: listing titles",
        "facebook token expired invalid_grant on ebay": "Josh: token fixes",
        "write an instagram caption for this product": "Josh: Instagram post",
        "price this pokemon trading card from sold comps": "Josh: card pricing",
        "should I buy this item from the store scan, is the spread profitable": "Josh: Penny Deals verdict",
    }
    for job, expected in jobs.items():
        picked = [p["name"] for p in ss.select_for(job)]
        assert expected in picked, (job, picked)


def test_every_skill_is_short_and_has_content():
    for s in seed_skills.SKILLS:
        assert 80 < len(s["text"]) < 1600, s["name"]
