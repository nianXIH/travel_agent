from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app

USER_APP_HTML = "旅游知识库"


def _client(tmp_path):
    app = create_app(Settings(_env_file=None, data_dir=tmp_path, llm_api_key=""))
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_user_app_page_is_served(tmp_path):
    async with _client(tmp_path) as client:
        response = await client.get("/app/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert USER_APP_HTML in response.text
    assert 'id="credential-gate"' in response.text


async def test_user_app_assets_are_served_and_wired(tmp_path):
    async with _client(tmp_path) as client:
        script = await client.get("/app/app.js")
        stylesheet = await client.get("/app/app.css")

    assert script.status_code == 200
    assert stylesheet.status_code == 200
    assert "/api/v1/chat/stream" in script.text
    assert "X-API-Key" in script.text
    assert "sessionStorage" in script.text
    assert "localStorage" not in script.text
    assert "X-Admin-Token" not in script.text
    assert "--accent" in stylesheet.text


async def test_root_links_to_user_app(tmp_path):
    async with _client(tmp_path) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json()["user_app"] == "/app/"
