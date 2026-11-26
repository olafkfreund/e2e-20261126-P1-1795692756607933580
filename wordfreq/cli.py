"""Command-line entry point and word frequency helpers for wordfreq."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_STDIN_PATH = "-"


def _validate_limits(min_len: int, top: int | None) -> None:
    """Raise ValueError when the `min_len` or `top` options are out of range."""
    if min_len < 1:
        msg = "min_len must be >= 1"
        raise ValueError(msg)
    if top is not None and top < 0:
        msg = "top must be >= 0"
        raise ValueError(msg)


def count_words(
    text: str,
    *,
    min_len: int = 1,
    ignore_case: bool = True,
    top: int | None = None,
) -> list[tuple[str, int]]:
    """Count word occurrences in *text* and return `(word, count)` pairs."""
    _validate_limits(min_len, top)
    tokens = _TOKEN_PATTERN.findall(text)
    if ignore_case:
        tokens = [token.lower() for token in tokens]
    counts = Counter(token for token in tokens if len(token) >= min_len)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top is not None:
        ranked = ranked[:top]
    return ranked


def format_table(rows: Sequence[tuple[str, int]]) -> str:
    """Format `(word, count)` pairs as tab-separated lines."""
    return "\n".join(f"{word}\t{count}" for word, count in rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordfreq",
        description="Deterministically count and rank words in text.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=_STDIN_PATH,
        help="file to read; '-' or omitted reads from stdin",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="show only the N most frequent words",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="ignore words shorter than N characters (default: 1)",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="count upper- and lower-case spellings separately",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON array of {word, count} objects",
    )
    return parser


def _read_input(path: str) -> str:
    if path == _STDIN_PATH:
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wordfreq CLI and return the process exit code."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse reports usage errors (already on stderr) via SystemExit.
        return exc.code if isinstance(exc.code, int) else 0

    try:
        _validate_limits(args.min_len, args.top)
    except ValueError as exc:
        sys.stderr.write(f"wordfreq: error: {exc}\n")
        return 2

    try:
        text = _read_input(args.path)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        sys.stderr.write(f"wordfreq: error: cannot read {args.path!r}: {reason}\n")
        return 2

    rows = count_words(
        text,
        min_len=args.min_len,
        ignore_case=not args.case_sensitive,
        top=args.top,
    )

    if args.json:
        payload = [{"word": word, "count": count} for word, count in rows]
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        table = format_table(rows)
        if table:
            sys.stdout.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
