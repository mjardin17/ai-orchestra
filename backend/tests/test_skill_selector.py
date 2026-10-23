import pytest
from app import skill_selector as ss, store

SKILLS = [
    {"id": "a", "name": "Ebay listing writer", "text": "Write eBay listing titles under 80 characters with brand, model and condition."},
    {"id": "b", "name": "Python debugging", "text": "Reproduce the bug, read the traceback, write a failing test, then fix the code."},
    {"id": "c", "name": "Instagram captions", "text": "Short caption, hook first line, hashtags, call to action for Instagram posts."},
]


def test_picks_the_skill_that_matches_the_job():
    out = ss.select_skills("write an ebay listing title for a Hot Wheels car", SKILLS)
    assert out[0]["id"] == "a"


def test_returns_explanation_of_why():
    out = ss.select_skills("fix this python traceback bug", SKILLS)
    assert out[0]["id"] == "b"
    assert "traceback" in out[0]["matched"] and out[0]["score"] > 0


def test_unrelated_job_selects_nothing():
    assert ss.select_skills("what is the capital of France", SKILLS) == []


def test_respects_k_and_orders_by_score():
    out = ss.select_skills("python bug and an instagram caption", SKILLS, k=1)
    assert len(out) == 1


def test_name_matches_outweigh_body_matches():
    skills = [{"id": "x", "name": "Pricing", "text": "unrelated words here"},
              {"id": "y", "name": "Other", "text": "pricing pricing"}]
    assert ss.select_skills("pricing", skills)[0]["id"] == "x"


def test_empty_inputs_are_safe():
    assert ss.select_skills("", SKILLS) == []
    assert ss.select_skills("anything", []) == []
    assert ss.select_skills(None, None) == []


def test_skill_block_is_capped_and_labelled():
    big = [{"id": "z", "name": "Big", "text": "ebay listing " * 2500}]
    block = ss.skill_block("ebay listing", big, limit=500)
    assert block.startswith("Skills picked for this job") and len(block) <= 700


def test_store_library_roundtrip_selects_saved_skill():
    store.DB.execute("delete from skills"); store.DB.commit()
    store.add_skill("Card grading", "Check centering, corners, edges and surface for trading card condition.")
    out = ss.select_for(" grade my trading card condition and corners")
    assert out and out[0]["name"] == "Card grading"


def test_for_job_skips_skills_already_pinned_to_the_workspace():
    import json
    store.DB.execute("delete from skills"); store.DB.commit()
    sid = store.add_skill("Ebay titles", "Write eBay listing titles with brand and model.")
    other = store.add_skill("Ebay photos", "eBay listing photos need a white background and good light.")
    wid = store.create_workspace("t")
    store.update_workspace(wid, skills=[sid])
    block, picked = ss.for_job("write ebay listing titles and photos", wid)
    assert [p["id"] for p in picked] == [other]
    assert "Ebay photos" in block and "Ebay titles" not in block


def test_for_job_without_workspace_still_picks():
    store.DB.execute("delete from skills"); store.DB.commit()
    store.add_skill("Card grading", "Check centering, corners, edges and surface for trading card condition.")
    block, picked = ss.for_job("grade the corners and edges of my trading card", None)
    assert picked and "Card grading" in block
