"""Execute the SQL analytics library against the SQLite database.

The SQL lives in ``sql/analytics_queries.sql`` rather than in Python strings so
it can be reviewed, diffed and handed to a BI/database reviewer as a real
artefact. This module parses the ``-- name:`` headers, runs each block, and
persists the results.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from .db_setup import get_connection

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^--\s*title:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class QueryBlock:
    name: str
    title: str
    sql: str


def parse_query_file(path: Path | None = None) -> list[QueryBlock]:
    """Split the SQL library into executable blocks keyed by ``-- name:``."""
    path = Path(path) if path is not None else config.SQL_QUERY_FILE
    if not path.exists():
        raise FileNotFoundError(f"SQL query file not found: {path}")

    text = path.read_text(encoding="utf-8")
    matches = list(_NAME_RE.finditer(text))
    blocks: list[QueryBlock] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]

        # Strip the trailing title/description metadata comments from the body
        # only at the top; SQL comments inside the statement are harmless.
        sql = _strip_leading_comments(body)
        if not sql.strip():
            continue

        title_match = _TITLE_RE.search(body)
        blocks.append(
            QueryBlock(
                name=match.group(1),
                title=title_match.group(1) if title_match else match.group(1),
                sql=sql.strip().rstrip(";").strip(),
            )
        )
    return blocks


def _strip_leading_comments(body: str) -> str:
    """Remove leading metadata comment lines from a SQL block body."""
    lines = body.splitlines()
    out: list[str] = []
    still_leading = True
    for line in lines:
        stripped = line.strip()
        if still_leading and (stripped.startswith("--") or stripped == ""):
            continue
        still_leading = False
        out.append(line)
    return "\n".join(out)


def run_query(name: str, db_path: Path | None = None, blocks: list[QueryBlock] | None = None) -> pd.DataFrame:
    """Run a single named SQL block and return its result as a DataFrame."""
    blocks = blocks if blocks is not None else parse_query_file()
    by_name = {b.name: b for b in blocks}
    if name not in by_name:
        raise KeyError(f"Unknown SQL block '{name}'. Available: {sorted(by_name)}")

    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(by_name[name].sql, conn)
    finally:
        conn.close()


def run_all_queries(
    db_path: Path | None = None,
    export_dir: Path | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute every block; optionally write one CSV per query."""
    export_dir = Path(export_dir) if export_dir is not None else config.POWERBI_EXPORTS / "sql"
    blocks = parse_query_file()
    conn = get_connection(db_path)
    results: dict[str, pd.DataFrame] = {}

    try:
        for block in blocks:
            try:
                frame = pd.read_sql_query(block.sql, conn)
            except Exception as exc:  # pragma: no cover - surfaced loudly
                raise RuntimeError(f"SQL block '{block.name}' failed: {exc}") from exc
            results[block.name] = frame
            logger.info(
                "SQL %-34s -> %s rows x %s cols", block.name, *frame.shape
            )
            if save:
                export_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(export_dir / f"{block.name}.csv", index=False)
    finally:
        conn.close()

    return results


def results_as_markdown(results: dict[str, pd.DataFrame], max_rows: int = 12) -> str:
    """Render the SQL results into a readable markdown appendix."""
    blocks = {b.name: b for b in parse_query_file()}
    sections: list[str] = []
    for name, frame in results.items():
        title = blocks[name].title if name in blocks else name
        sections.append(f"### {title}\n\n_Query block: `{name}`_\n")
        view = frame.head(max_rows).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
        headers = [str(c) for c in view.columns]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
        if len(frame) > max_rows:
            lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
