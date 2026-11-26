"""Unit and CLI tests for the wordfreq package."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wordfreq.cli import count_words, format_table, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(
    args: list[str], stdin_text: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run `python -m wordfreq` in a subprocess from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        check=False,
    )


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_cli_top_and_min_len(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


# --- count_words: tokenisation, case, filtering, ordering -----------------


def test_count_words_tokens_are_ascii_alphanumeric() -> None:
    # Punctuation, apostrophes and non-ASCII letters split or drop tokens.
    assert count_words("don't v2.0 café") == [
        ("0", 1),
        ("caf", 1),
        ("don", 1),
        ("t", 1),
        ("v2", 1),
    ]


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Apple apple APPLE") == [("apple", 3)]


def test_count_words_preserves_case_when_ignore_case_false() -> None:
    assert count_words("Apple apple APPLE", ignore_case=False) == [
        ("APPLE", 1),
        ("Apple", 1),
        ("apple", 1),
    ]


def test_count_words_min_len_filters_short_tokens() -> None:
    assert count_words("a bb ccc bb", min_len=2) == [("bb", 2), ("ccc", 1)]


def test_count_words_tie_breaks_alphabetically_not_by_first_seen() -> None:
    # Every word appears once, so the alphabetical order must win over the
    # order in which the words appear in the text.
    assert count_words("zebra mango apple") == [
        ("apple", 1),
        ("mango", 1),
        ("zebra", 1),
    ]


def test_count_words_tie_break_is_deterministic_across_counts() -> None:
    text = "pear apple banana apple banana"
    assert count_words(text) == [("apple", 2), ("banana", 2), ("pear", 1)]


def test_count_words_top_limits_results() -> None:
    assert count_words("c b a c b c", top=2) == [("c", 3), ("b", 2)]


def test_count_words_top_zero_returns_empty() -> None:
    assert count_words("a a b", top=0) == []


def test_count_words_empty_text_returns_empty_list() -> None:
    assert count_words("") == []


def test_count_words_rejects_min_len_below_one() -> None:
    with pytest.raises(ValueError, match="min_len must be >= 1"):
        count_words("a", min_len=0)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match="top must be >= 0"):
        count_words("a", top=-1)


# --- format_table ---------------------------------------------------------


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([]) == ""


# --- main: input sources --------------------------------------------------


def test_main_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("red blue red\n"))
    rc = main([])
    assert rc == 0
    assert capsys.readouterr().out == "red\t2\nblue\t1\n"


def test_main_reads_stdin_for_dash_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("red blue red\n"))
    rc = main(["-"])
    assert rc == 0
    assert capsys.readouterr().out == "red\t2\nblue\t1\n"


def test_main_case_sensitive_counts_spellings_separately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "case.txt"
    sample.write_text("Apple apple APPLE", encoding="utf-8")
    rc = main([str(sample), "--case-sensitive"])
    assert rc == 0
    assert capsys.readouterr().out == "APPLE\t1\nApple\t1\napple\t1\n"


def test_main_case_insensitive_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "case.txt"
    sample.write_text("Apple apple APPLE", encoding="utf-8")
    rc = main([str(sample)])
    assert rc == 0
    assert capsys.readouterr().out == "apple\t3\n"


# --- main: output formatting ----------------------------------------------


def test_main_tsv_output_ends_with_single_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("apple apple banana", encoding="utf-8")
    rc = main([str(sample)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "apple\t2\nbanana\t1\n"
    assert out.endswith("\n")
    assert not out.endswith("\n\n")


def test_main_empty_tsv_output_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "empty.txt"
    sample.write_text("   \n\t", encoding="utf-8")
    rc = main([str(sample)])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_json_output_ends_with_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("apple apple banana", encoding="utf-8")
    rc = main([str(sample), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
    assert json.loads(out) == [
        {"word": "apple", "count": 2},
        {"word": "banana", "count": 1},
    ]
    assert out == json.dumps(json.loads(out), indent=2) + "\n"


def test_main_json_empty_input_is_empty_array_with_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "empty.txt"
    sample.write_text("", encoding="utf-8")
    rc = main([str(sample), "--json"])
    assert rc == 0
    assert capsys.readouterr().out == "[]\n"


def test_main_json_respects_top_and_min_len(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("to be or not to be that is the question")
    )
    rc = main(["--json", "--min-len", "3", "--top", "2"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == [
        {"word": "not", "count": 1},
        {"word": "question", "count": 1},
    ]


# --- main: argument validation and error exit codes -----------------------


def test_main_rejects_min_len_below_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("apple", encoding="utf-8")
    rc = main([str(sample), "--min-len", "0"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "min_len must be >= 1" in captured.err


def test_main_rejects_negative_top(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("apple", encoding="utf-8")
    rc = main([str(sample), "--top", "-1"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "top must be >= 0" in captured.err


def test_main_rejects_non_integer_option_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("apple", encoding="utf-8")
    rc = main([str(sample), "--top", "many"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


def test_main_rejects_unknown_option(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--no-such-option"])
    assert rc == 2
    assert capsys.readouterr().err != ""


def test_main_missing_file_returns_2_with_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.txt"
    rc = main([str(missing)])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read" in captured.err
    assert str(missing) in captured.err


# --- python -m wordfreq ---------------------------------------------------


def test_python_m_wordfreq_reads_stdin() -> None:
    result = _run_module([], stdin_text="red blue red\n")
    assert result.returncode == 0
    assert result.stdout == "red\t2\nblue\t1\n"
    assert result.stderr == ""


def test_python_m_wordfreq_json_output_ends_with_newline() -> None:
    result = _run_module(["--json"], stdin_text="red blue red\n")
    assert result.returncode == 0
    assert result.stdout.endswith("\n")
    assert not result.stdout.endswith("\n\n")
    assert json.loads(result.stdout) == [
        {"word": "red", "count": 2},
        {"word": "blue", "count": 1},
    ]


def test_python_m_wordfreq_empty_tsv_prints_nothing() -> None:
    result = _run_module([], stdin_text="")
    assert result.returncode == 0
    assert result.stdout == ""


def test_python_m_wordfreq_missing_file_exits_2() -> None:
    result = _run_module(["definitely-missing-file.txt"])
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr != ""


def test_python_m_wordfreq_invalid_argument_exits_2() -> None:
    result = _run_module(["--min-len", "0"], stdin_text="apple\n")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "min_len must be >= 1" in result.stderr
