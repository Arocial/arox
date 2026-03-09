import pytest


@pytest.fixture(autouse=True)
def mock_home_env(monkeypatch, tmp_path):
    """
    Mock the HOME directory for all tests to prevent exposing or modifying
    the real home directory.
    """
    mock_home = tmp_path / "mock_home"
    mock_home.mkdir()

    monkeypatch.setenv("HOME", str(mock_home))
    monkeypatch.setenv("USERPROFILE", str(mock_home))
