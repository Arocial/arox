from pathlib import Path

from arox.core.skills import build_skill_catalog, discover_skills


def test_discover_skills_empty(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skills = discover_skills(workspace)
    assert skills == {}


def test_discover_skills_valid(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".arox" / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: test_skill\ndescription: A test skill\n---\nSkill content here",
        encoding="utf-8",
    )

    skills = discover_skills(workspace)
    assert "test_skill" in skills
    assert skills["test_skill"]["name"] == "test_skill"
    assert skills["test_skill"]["description"] == "A test skill"
    assert skills["test_skill"]["location"] == str(skill_file.absolute())


def test_discover_skills_malformed_yaml_fixed(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".arox" / "skills" / "malformed_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: malformed_skill\n"
        "description: A skill with a colon: in description\n"
        "---\n"
        "Content",
        encoding="utf-8",
    )

    skills = discover_skills(workspace)
    assert "malformed_skill" in skills
    assert (
        skills["malformed_skill"]["description"]
        == "A skill with a colon: in description"
    )


def test_discover_skills_missing_metadata(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".arox" / "skills" / "missing_meta"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: missing_meta\n---\nContent", encoding="utf-8")

    skills = discover_skills(workspace)
    assert "missing_meta" not in skills


def test_build_skill_catalog():
    assert build_skill_catalog({}) == ""

    skills = {
        "test_skill": {
            "name": "test_skill",
            "description": "A test skill",
            "location": "/path/to/SKILL.md",
        }
    }

    catalog = build_skill_catalog(skills)
    assert "<available_skills>" in catalog
    assert "<name>test_skill</name>" in catalog
    assert "<description>A test skill</description>" in catalog
    assert "<location>/path/to/SKILL.md</location>" in catalog
