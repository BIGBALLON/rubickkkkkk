"""CLI entry: ``python -m rubick_backend <command> [args]``.

A thin wrapper around the ingest pipeline and a vector-search smoke,
suitable for dogfooding before the FastAPI routes are wired up.

Examples::

    # ingest a folder
    python -m rubick_backend ingest ~/notes/

    # query
    python -m rubick_backend search "red planet"

The data directory defaults to ``~/Library/Application Support/Rubick/``;
override with ``RUBICK_DATA_DIR=/tmp/x`` for ephemeral smokes.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .ingest import ingest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rubick_backend")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable INFO logging from rubick_backend.*",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="ingest text files into LanceDB")
    ingest.add_argument("path", help="file or directory to ingest")

    search = sub.add_parser("search", help="vector-search the LanceDB index")
    search.add_argument("query", help="natural-language query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument(
        "--modality",
        default=None,
        help="filter by modality (e.g. text)",
    )
    search.add_argument(
        "--path-prefix",
        default=None,
        help="keep only docs whose canonical path starts with this prefix",
    )
    search.add_argument(
        "--mtime-after",
        default=None,
        type=_parse_mtime_arg,
        help=("keep only docs with mtime >= this date (YYYY-MM-DD or POSIX epoch seconds)"),
    )
    search.add_argument(
        "--mtime-before",
        default=None,
        type=_parse_mtime_arg,
        help=("keep only docs with mtime <= this date (YYYY-MM-DD or POSIX epoch seconds)"),
    )
    return parser


def _parse_mtime_arg(value: str) -> int:
    """Accept either a POSIX epoch second integer or a ``YYYY-MM-DD``
    calendar date. The date form is parsed in the local timezone and
    snapped to the start of the day — matches what users mean by
    "after Jan 1" without forcing them to think in UTC.
    """
    import datetime

    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        date = datetime.datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected POSIX epoch seconds or YYYY-MM-DD, got {value!r}"
        ) from e
    return int(date.timestamp())


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def cmd_ingest(args) -> int:
    stats = ingest_path(args.path)
    print(f"ingested {stats['files']} files, {stats['chunks']} chunks; skipped {stats['skipped']}")
    return 0


def cmd_search(args) -> int:
    """Hybrid search the index — same code path the FastAPI route uses.

    Doc-level folded results: vector + BM25 + RRF when text is provided.
    RRF fusion, default-hide rejected rows. The CLI is mostly a
    dogfood / debug surface, so we print all three sub-scores
    alongside the headline RRF.
    """
    from .embed import embed_query
    from .retrieve import hybrid_search

    qvec = embed_query(args.query)
    results = hybrid_search(
        qvec=qvec,
        qtext=args.query,
        doc_limit=args.limit,
        modality=args.modality,
        path_prefix=args.path_prefix,
        mtime_after=args.mtime_after,
        mtime_before=args.mtime_before,
    )
    if not results:
        print("no results")
        return 0
    for r in results:
        path = r.file_paths[0] if r.file_paths else "?"
        preview = (r.raw_text or "").replace("\n", " ")[:120]
        sim_part = f"sim={r.similarity:+.3f}"
        bm25_part = f"bm25={r.score_bm25:.2f}" if r.score_bm25 is not None else "bm25=  -  "
        hit_part = f"x{r.hit_count}" if r.hit_count > 1 else "   "
        print(
            f"[rrf {r.score_rrf:.4f} | {sim_part} | {bm25_part} | {hit_part}] "
            f"{r.modality:>16}  {path}"
        )
        if preview:
            print(f"                          {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if args.cmd == "ingest":
        return cmd_ingest(args)
    if args.cmd == "search":
        return cmd_search(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
