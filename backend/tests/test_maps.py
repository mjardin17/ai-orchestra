from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def test_empire_os_map_is_served():
    r = c.get("/maps/empire-os.html")
    assert r.status_code == 200
    assert "Empire OS" in r.text


def test_bosslister_map_is_served():
    r = c.get("/maps/bosslister.html")
    assert r.status_code == 200
    assert "BossLister" in r.text


def test_unknown_map_is_404():
    assert c.get("/maps/nope.html").status_code == 404


def test_maps_cannot_escape_the_folder():
    assert c.get("/maps/..%2Findex.html").status_code in (400, 404)
