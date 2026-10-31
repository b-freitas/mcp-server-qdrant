import pytest


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
