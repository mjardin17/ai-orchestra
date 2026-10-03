from fastapi.testclient import TestClient
from app.main import app
from app import store

c = TestClient(app)


def setup_function():
    store.DB.execute("delete from lessons"); store.DB.commit()


def test_add_approve_and_list_lesson():
    lid = c.post("/api/lessons", json={"rule": "Don't retry a rejected token", "agent": "*"}).json()["id"]
    assert c.put(f"/api/lessons/{lid}", json={"status": "active"}).status_code == 200
    assert [l["id"] for l in c.get("/api/lessons?status=active").json()] == [lid]


def test_bad_status_and_missing_lesson():
    lid = c.post("/api/lessons", json={"rule": "x"}).json()["id"]
    assert c.put(f"/api/lessons/{lid}", json={"status": "nope"}).status_code == 400
    assert c.put("/api/lessons/missing", json={"status": "active"}).status_code == 404


def test_empty_rule_rejected():
    assert c.post("/api/lessons", json={"rule": "  "}).status_code == 400


def test_skill_suggest_endpoint_returns_matches():
    store.DB.execute("delete from skills"); store.DB.commit()
    store.add_skill("Ebay titles", "Write eBay listing titles with brand and model.")
    r = c.post("/api/skills/suggest", json={"prompt": "write an ebay listing title"})
    assert r.status_code == 200 and r.json()[0]["name"] == "Ebay titles"


def test_skill_suggest_rejects_empty_prompt():
    assert c.post("/api/skills/suggest", json={"prompt": "  "}).status_code == 400
