from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.schemas import AssetStatus, ProjectCreate, utc_now


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Repository:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.setup()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def setup(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_text TEXT NOT NULL,
            culture TEXT NOT NULL,
            locality TEXT NOT NULL,
            setting TEXT NOT NULL,
            period TEXT NOT NULL,
            output_script TEXT NOT NULL,
            thread_id TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            interrupt_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS revisions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            version INTEGER NOT NULL,
            parent_id TEXT,
            payload_json TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            approved_at TEXT,
            UNIQUE(project_id, kind, version)
        );
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            gate TEXT NOT NULL,
            revision_id TEXT,
            override_reason TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            canonical_id TEXT NOT NULL,
            scene_id TEXT,
            prompt TEXT NOT NULL,
            dependency_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            path TEXT,
            error TEXT,
            attempt INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project_id, kind, canonical_id, dependency_hash)
        );
        CREATE TABLE IF NOT EXISTS model_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            cache_key TEXT NOT NULL UNIQUE,
            output_json TEXT,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idempotency (
            scope TEXT NOT NULL,
            key TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(scope, key)
        );
        """
        with self.connect() as connection:
            connection.executescript(schema)

    def create_project(self, request: ProjectCreate) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO projects
                (id,title,source_text,culture,locality,setting,period,output_script,thread_id,stage,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    project_id, request.title, request.source_text, request.culture,
                    request.locality, request.setting, request.period, request.output_script,
                    project_id, "created", "active", now, now,
                ),
            )
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,title,culture,locality,setting,stage,status,created_at,updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError(f"Project {project_id} was not found")
        result = dict(row)
        interrupt_json = result.pop("interrupt_json", None)
        result["interrupt"] = json.loads(interrupt_json) if interrupt_json else None
        return result

    def set_stage(self, project_id: str, stage: str, *, status: str = "active", interrupt: dict[str, Any] | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE projects SET stage=?, status=?, interrupt_json=?, updated_at=? WHERE id=?",
                (stage, status, json.dumps(interrupt, ensure_ascii=False) if interrupt else None, utc_now(), project_id),
            )

    def create_revision(self, project_id: str, kind: str, payload: Any, parent_id: str | None = None) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        with self._lock, self.connect() as connection:
            version = connection.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM revisions WHERE project_id=? AND kind=?",
                (project_id, kind),
            ).fetchone()[0]
            revision_id = str(uuid.uuid4())
            encoded = canonical_json(payload)
            now = utc_now()
            connection.execute(
                """INSERT INTO revisions
                (id,project_id,kind,version,parent_id,payload_json,sha256,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (revision_id, project_id, kind, version, parent_id, encoded, content_hash(encoded), now),
            )
        return self.get_revision(revision_id)

    def get_revision(self, revision_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM revisions WHERE id=?", (revision_id,)).fetchone()
        if not row:
            raise KeyError(f"Revision {revision_id} was not found")
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def latest_revision(self, project_id: str, kind: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM revisions WHERE project_id=? AND kind=? ORDER BY version DESC LIMIT 1",
                (project_id, kind),
            ).fetchone()
        return self.get_revision(row["id"]) if row else None

    def list_latest_revisions(self, project_id: str) -> dict[str, dict[str, Any]]:
        query = """
        SELECT r.id FROM revisions r
        JOIN (SELECT kind, MAX(version) AS version FROM revisions WHERE project_id=? GROUP BY kind) latest
        ON r.kind=latest.kind AND r.version=latest.version
        WHERE r.project_id=?
        """
        with self.connect() as connection:
            rows = connection.execute(query, (project_id, project_id)).fetchall()
        return {rev["kind"]: rev for rev in (self.get_revision(row["id"]) for row in rows)}

    def approve(self, project_id: str, gate: str, revision_id: str | None, override_reason: str | None = None) -> dict[str, Any]:
        now = utc_now()
        approval = {
            "id": str(uuid.uuid4()), "project_id": project_id, "gate": gate,
            "revision_id": revision_id, "override_reason": override_reason, "created_at": now,
        }
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO approvals (id,project_id,gate,revision_id,override_reason,created_at) VALUES (?,?,?,?,?,?)",
                tuple(approval.values()),
            )
            if revision_id:
                connection.execute("UPDATE revisions SET approved_at=? WHERE id=?", (now, revision_id))
        return approval

    def list_approvals(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM approvals WHERE project_id=? ORDER BY created_at", (project_id,)).fetchall()
        return [dict(row) for row in rows]

    def upsert_asset(
        self, project_id: str, kind: str, canonical_id: str, prompt: str,
        dependency_hash: str, scene_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        asset_id = str(uuid.uuid5(uuid.UUID(project_id), f"{kind}:{canonical_id}:{dependency_hash}"))
        with self.connect() as connection:
            connection.execute(
                """UPDATE assets SET status=?, updated_at=?
                WHERE project_id=? AND kind=? AND canonical_id=? AND dependency_hash<>? AND status<>?""",
                (AssetStatus.invalidated, now, project_id, kind, canonical_id, dependency_hash, AssetStatus.invalidated),
            )
            connection.execute(
                """INSERT OR IGNORE INTO assets
                (id,project_id,kind,canonical_id,scene_id,prompt,dependency_hash,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (asset_id, project_id, kind, canonical_id, scene_id, prompt, dependency_hash, AssetStatus.pending, now, now),
            )
        return self.get_asset(asset_id)

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if not row:
            raise KeyError(f"Asset {asset_id} was not found")
        return dict(row)

    def list_assets(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM assets WHERE project_id=? ORDER BY kind,created_at", (project_id,)).fetchall()
        return [dict(row) for row in rows]

    def update_asset(self, asset_id: str, status: AssetStatus | str, *, path: str | None = None, error: str | None = None, increment_attempt: bool = False) -> dict[str, Any]:
        attempt_sql = ", attempt=attempt+1" if increment_attempt else ""
        with self.connect() as connection:
            connection.execute(
                f"UPDATE assets SET status=?, path=COALESCE(?,path), error=?, updated_at=?{attempt_sql} WHERE id=?",
                (str(status), path, error, utc_now(), asset_id),
            )
        return self.get_asset(asset_id)

    def invalidate_assets(self, project_id: str, canonical_ids: list[str]) -> None:
        if not canonical_ids:
            return
        placeholders = ",".join("?" for _ in canonical_ids)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE assets SET status=?, updated_at=? WHERE project_id=? AND canonical_id IN ({placeholders})",
                (AssetStatus.invalidated, utc_now(), project_id, *canonical_ids),
            )

    def get_cached_model_run(self, cache_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_runs WHERE cache_key=? AND status='success'", (cache_key,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["output"] = json.loads(result.pop("output_json"))
        return result

    def save_model_run(self, **values: Any) -> dict[str, Any]:
        values.setdefault("id", str(uuid.uuid4()))
        values.setdefault("created_at", utc_now())
        values["output_json"] = canonical_json(values.pop("output")) if "output" in values else None
        columns = [
            "id", "project_id", "operation", "provider", "model", "prompt_hash", "input_hash",
            "cache_key", "output_json", "status", "latency_ms", "input_tokens", "output_tokens", "error", "created_at",
        ]
        with self.connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO model_runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [values.get(column) for column in columns],
            )
        return values

    def list_model_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,operation,provider,model,prompt_hash,input_hash,status,latency_ms,input_tokens,output_tokens,error,created_at FROM model_runs WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_idempotent(self, scope: str, key: str | None) -> Any | None:
        if not key:
            return None
        with self.connect() as connection:
            row = connection.execute("SELECT response_json FROM idempotency WHERE scope=? AND key=?", (scope, key)).fetchone()
        return json.loads(row[0]) if row else None

    def save_idempotent(self, scope: str, key: str | None, response: Any) -> None:
        if not key:
            return
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO idempotency (scope,key,response_json,created_at) VALUES (?,?,?,?)",
                (scope, key, canonical_json(response), utc_now()),
            )
