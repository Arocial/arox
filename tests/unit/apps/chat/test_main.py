import logging
from unittest.mock import patch

from arox.apps.chat.main import _configure_logging, _create_argument_parser


def test_debug_flag_is_disabled_by_default():
    args = _create_argument_parser().parse_args([])

    assert args.debug is False


def test_debug_flag_can_be_enabled():
    args = _create_argument_parser().parse_args(["--debug"])

    assert args.debug is True


def test_debug_logging_uses_debug_level_for_text_ui(tmp_path):
    with (
        patch("platformdirs.user_log_dir", return_value=str(tmp_path)),
        patch("logging.basicConfig") as basic_config,
    ):
        _configure_logging("text", debug=True)

    basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=tmp_path / "agents.log",
        filemode="a",
    )


def test_default_console_logging_uses_info_level():
    with patch("logging.basicConfig") as basic_config:
        _configure_logging("vercel_ai")

    basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
