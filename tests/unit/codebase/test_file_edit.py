import random
import string
import tempfile
from pathlib import Path

import pytest

from arox.plugins.file import FilePlugin

original_content = """import yaml
import os
import sys
from pathlib import Path

def test():
    pass"""


class MockAgent:
    def __init__(self, workspace):
        self.workspace = workspace
        self.agent_io = None
        self._capabilities = {}

    def provide_capability(self, capability, provider):
        self._capabilities[capability] = provider

    def get_capability(self, capability, default=None):
        return self._capabilities.get(capability, default)


class TestFileEdit:
    @classmethod
    def setup_class(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls.temp_dir.name)
        cls.agent = MockAgent(cls.workspace)
        cls.tool = FilePlugin(cls.agent)  # type: ignore

    @classmethod
    def teardown_class(cls):
        cls.temp_dir.cleanup()

    @pytest.mark.asyncio
    async def test_write_to_file_new_file(self):
        """Test writing to a new file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            content = "Hello, World!"

            result = await self.tool.write_to_file(str(file_path), content)

            assert "Successfully wrote to" in result
            assert file_path.exists()
            assert file_path.read_text() == content

    @pytest.mark.asyncio
    async def test_write_to_file_overwrite(self):
        """Test overwriting an existing file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("Original content")

            new_content = "New content"
            result = await self.tool.write_to_file(str(file_path), new_content)

            assert "Successfully wrote to" in result
            assert file_path.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_to_file_create_directories(self):
        """Test creating directories when they don't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "subdir" / "test.txt"
            content = "Test content"

            result = await self.tool.write_to_file(str(file_path), content)

            assert "Successfully wrote to" in result
            assert file_path.exists()
            assert file_path.read_text() == content

    @pytest.mark.asyncio
    async def test_replace_in_file_simple(self):
        """Test simple content replacement"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.py"
            file_path.write_text(original_content)

            old_str = "import yaml"
            new_str = "import json"

            result = await self.tool.replace_in_file(str(file_path), old_str, new_str)

            assert "Successfully updated" in result
            updated_content = file_path.read_text()
            assert "import json" in updated_content
            assert "import yaml" not in updated_content

    @pytest.mark.asyncio
    async def test_replace_in_file_with_placeholder(self):
        """Test replacement with ...omit lines... placeholder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.py"
            file_path.write_text(original_content)

            old_str = """import yaml
...omit lines...
from pathlib import Path"""
            new_str = """import json

from pathlib import Path"""

            result = await self.tool.replace_in_file(str(file_path), old_str, new_str)

            assert "Successfully updated" in result
            updated_content = file_path.read_text()
            assert "import json" in updated_content
            assert "import yaml" not in updated_content
            assert "def test():" in updated_content  # Should preserve content after

    @pytest.mark.asyncio
    async def test_replace_in_file_nonexistent(self):
        """Test replacement on non-existent file"""
        result = await self.tool.replace_in_file("/nonexistent/file.py", "old", "new")
        assert "File not found" in result

    @pytest.mark.asyncio
    async def test_replace_in_file_fuzzy(self):
        """Test fuzzy content replacement with minor whitespace/case differences"""
        with tempfile.TemporaryDirectory() as temp_dir:
            characters = string.ascii_letters + string.digits
            file_path = Path(temp_dir) / "test.py"
            random_string = original_content
            for _ in range(10):
                length = random.randint(0, 100)
                line = "".join(random.choice(characters) for _ in range(length))
                random_string = f"{random_string}\n{line}"

            lines = random_string.splitlines()
            old_strs = [
                "\n".join(lines[1:-1]).replace("import sys", "import   sys"),
                "\n".join(lines[1:-1]).replace("import sys", "importsys"),
                "\n".join(lines[1:-2]) + "\n  " + lines[-2],
                "\n".join(lines[1:-2]) + lines[-2],
            ]
            new_str = "replaced\n"

            for old_str in old_strs:
                file_path.write_text(random_string)
                result = await self.tool.replace_in_file(
                    str(file_path), old_str, new_str
                )

                assert "Successfully updated" in result
                updated_content = file_path.read_text()
                assert f"{lines[0]}\nreplaced\n{lines[-1]}" == updated_content
