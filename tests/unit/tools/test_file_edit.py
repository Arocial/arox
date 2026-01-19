import tempfile
from pathlib import Path

import pytest
from arox.tools.file_edit import FileEdit


class TestFileEdit:
    @classmethod
    def setup_class(cls):
        cls.tool = FileEdit(None, None)

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
            original_content = """import yaml
from pathlib import Path

def test():
    pass"""
            file_path.write_text(original_content)

            diff = """
<<<<<<< SEARCH
import yaml
=======
import json
>>>>>>> REPLACE
            """

            result = await self.tool.replace_in_file(str(file_path), diff)

            assert "Successfully updated" in result
            updated_content = file_path.read_text()
            assert "import json" in updated_content
            assert "import yaml" not in updated_content

            diff2 = """
<<<<<<< SEARCH
import yaml
=======
>>>>>>> REPLACE
            """
            result = await self.tool.replace_in_file(str(file_path), diff2)

            assert "Successfully updated" in result
            updated_content = file_path.read_text()
            assert "import yaml" not in updated_content

    @pytest.mark.asyncio
    async def test_replace_in_file_with_placeholder(self):
        """Test replacement with ...existing code... placeholder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.py"
            original_content = """import yaml
import os
import sys
from pathlib import Path

def test():
    pass"""
            file_path.write_text(original_content)

            diff = """<<<<<<< SEARCH
import yaml
# ...existing code...
from pathlib import Path
=======
import json

from pathlib import Path
>>>>>>> REPLACE"""

            result = await self.tool.replace_in_file(str(file_path), diff)

            assert "Successfully updated" in result
            updated_content = file_path.read_text()
            assert "import json" in updated_content
            assert "import yaml" not in updated_content
            assert "def test():" in updated_content  # Should preserve content after

    @pytest.mark.asyncio
    async def test_replace_in_file_nonexistent(self):
        """Test replacement on non-existent file"""
        result = await self.tool.replace_in_file("/nonexistent/file.py", "some diff")
        assert "File not found" in result
