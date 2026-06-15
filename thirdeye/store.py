from __future__ import annotations

import json
import os
import sqlite3
import shutil
from pathlib import Path
from typing import Any, Iterable

from thirdeye.hashing import hash_file


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (record_type, record_id)
);
CREATE INDEX IF NOT EXISTS idx_records_project
ON records(project_id, record_type, created_at);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (run_id, metric_id)
);
"""


def default_home() -> Path:
    return Path(os.environ.get("THIRDEYE_HOME", ".thirdeye")).resolve()


class EvidenceStore:
    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home).resolve() if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db_path = self.home / "evidence.sqlite3"
        self.artifact_dir = self.home / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def put(
        self,
        record_type: str,
        record_id: str,
        project_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO records(record_type, record_id, project_id, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(record_type, record_id) DO UPDATE SET
                    project_id=excluded.project_id,
                    payload_json=excluded.payload_json
                """,
                (record_type, record_id, project_id, json.dumps(payload, sort_keys=True)),
            )

    def get(self, record_type: str, record_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM records WHERE record_type=? AND record_id=?",
                (record_type, record_id),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list(self, record_type: str, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM records
                WHERE record_type=? AND project_id=?
                ORDER BY created_at, record_id
                """,
                (record_type, project_id),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_all(self, record_type: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM records
                WHERE record_type=?
                ORDER BY created_at, record_id
                """,
                (record_type,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def put_metrics(self, run_id: str, metrics: Iterable[dict[str, Any]]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO metrics(run_id, metric_id, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id, metric_id) DO UPDATE SET
                    payload_json=excluded.payload_json
                """,
                [
                    (run_id, metric["metric_id"], json.dumps(metric, sort_keys=True))
                    for metric in metrics
                ],
            )

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM metrics WHERE run_id=? ORDER BY metric_id",
                (run_id,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def add_artifact(
        self,
        *,
        project_id: str,
        run_id: str,
        source: str | Path,
        kind: str = "artifact",
    ) -> dict[str, Any]:
        path = Path(source)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        digest = hash_file(path)
        target = self.artifact_dir / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        payload = {
            "artifact_id": digest,
            "project_id": project_id,
            "run_id": run_id,
            "kind": kind,
            "source_name": path.name,
            "path": str(target),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
        self.put("artifact", f"{run_id}:{digest}", project_id, payload)
        return payload
