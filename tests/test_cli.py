"""The command-line tool starts and every command's options parse (catches a helper registered as a command)."""

from typer.testing import CliRunner

from bluecarbon.cli import app


def test_cli_help_for_every_command():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("fetch", "chips", "train", "evaluate", "predict", "report", "scene", "case-study", "export-demo"):
        r = CliRunner().invoke(app, [cmd, "--help"])
        assert r.exit_code == 0, (cmd, r.output)
