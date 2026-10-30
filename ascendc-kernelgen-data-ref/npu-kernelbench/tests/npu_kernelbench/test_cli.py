"""Tests for CLI entry point."""

from click.testing import CliRunner

from npu_kernelbench.cli.main import cli


class TestCli:
    def test_missing_args_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Group without args shows usage error
        assert result.exit_code != 0

    def test_help_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "eval" in result.output.lower()

    def test_missing_solution(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "/nonexistent/problem_dir"])
        assert result.exit_code != 0

    def test_eval_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "solution" in result.output.lower()