from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.cultures.models import CultureProfileSnapshot, TargetSelection
from backend.cultures.registry import culture_registry
from backend.schemas import AssetStatus, ProjectCreate, utc_now


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Repository:
    SCHEMA_VERSION = "3"

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
        migrate_from_v2 = False
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
            ).fetchone()
            version_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
            ).fetchone()
            if existing and not version_table:
                raise RuntimeError(
                    "This database predates culture profiles. Archive the current DATA_DIR and start with an empty "
                    "directory; existing projects are intentionally not migrated."
                )
            if version_table:
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key='schema_version'"
                ).fetchone()
                migrate_from_v2 = bool(row and row[0] == "2")
                if not row or (row[0] != self.SCHEMA_VERSION and not migrate_from_v2):
                    raise RuntimeError(
                        f"Unsupported application database schema {row[0] if row else 'unknown'}; "
                        f"expected {self.SCHEMA_VERSION}."
                    )
        schema = """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_text TEXT NOT NULL,
            culture_id TEXT NOT NULL,
            culture_display_name TEXT NOT NULL,
            profile_version TEXT NOT NULL,
            profile_hash TEXT NOT NULL,
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
            culture_id TEXT,
            profile_hash TEXT,
            brief_revision_id TEXT,
            output_json TEXT,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            operation_metadata_json TEXT,
            input_json TEXT,
            input_preview_json TEXT,
            output_preview_json TEXT,
            usage_json TEXT,
            cost_json TEXT,
            estimated_cost_usd REAL,
            pricing_catalog_version TEXT,
            cost_basis TEXT,
            cost_calculation_basis TEXT,
            model_version TEXT,
            response_id TEXT,
            finish_reason TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            cache_hit_count INTEGER NOT NULL DEFAULT 0,
            langfuse_trace_id TEXT,
            langfuse_observation_id TEXT,
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_call_attempts (
            id TEXT PRIMARY KEY,
            model_run_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            invocation_id TEXT,
            operation TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            latency_ms INTEGER,
            provider_latency_ms INTEGER,
            parse_latency_ms INTEGER,
            io_latency_ms INTEGER,
            usage_json TEXT,
            cost_json TEXT,
            estimated_cost_usd REAL,
            model_version TEXT,
            response_id TEXT,
            finish_reason TEXT,
            error_type TEXT,
            error TEXT,
            langfuse_trace_id TEXT,
            langfuse_observation_id TEXT,
            UNIQUE(model_run_id, attempt)
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
            if migrate_from_v2:
                self._migrate_model_runs_v2_to_v3(connection)
            connection.execute(
                "INSERT OR REPLACE INTO schema_metadata (key,value) VALUES ('schema_version',?)",
                (self.SCHEMA_VERSION,),
            )

    @staticmethod
    def _migrate_model_runs_v2_to_v3(connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(model_runs)").fetchall()}
        additions = {
            "total_tokens": "INTEGER",
            "operation_metadata_json": "TEXT",
            "input_json": "TEXT",
            "input_preview_json": "TEXT",
            "output_preview_json": "TEXT",
            "usage_json": "TEXT",
            "cost_json": "TEXT",
            "estimated_cost_usd": "REAL",
            "pricing_catalog_version": "TEXT",
            "cost_basis": "TEXT",
            "cost_calculation_basis": "TEXT",
            "model_version": "TEXT",
            "response_id": "TEXT",
            "finish_reason": "TEXT",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "cache_hit_count": "INTEGER NOT NULL DEFAULT 0",
            "langfuse_trace_id": "TEXT",
            "langfuse_observation_id": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE model_runs ADD COLUMN {name} {declaration}")

    def create_project(
        self, request: ProjectCreate, snapshot: CultureProfileSnapshot | None = None,
    ) -> dict[str, Any]:
        selection = TargetSelection(
            culture_id=request.culture_id, locality=request.locality, setting=request.setting,
            period=request.period, output_script=request.output_script,
        )
        snapshot = snapshot or culture_registry.validate_selection(selection)
        profile = snapshot.profile
        if not selection.locality.strip():
            raise ValueError("Exact locality is required")
        if profile.culture_id != selection.culture_id:
            raise ValueError("Culture profile snapshot does not match the requested culture")
        if selection.setting not in profile.supported_settings:
            raise ValueError(f"Unsupported setting for {profile.display_name}: {selection.setting}")
        if selection.period not in {item.id for item in profile.supported_periods}:
            raise ValueError(f"Unsupported period for {profile.display_name}: {selection.period}")
        if selection.output_script not in {item.id for item in profile.supported_scripts}:
            raise ValueError(f"Unsupported output script for {profile.display_name}: {selection.output_script}")
        project_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO projects
                (id,title,source_text,culture_id,culture_display_name,profile_version,profile_hash,
                 locality,setting,period,output_script,thread_id,stage,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    project_id, request.title, request.source_text, request.culture_id,
                    snapshot.profile.display_name, snapshot.profile.version, snapshot.profile_hash,
                    request.locality, request.setting, request.period, request.output_script,
                    project_id, "created", "active", now, now,
                ),
            )
        self.create_revision(project_id, "culture_profile_snapshot", snapshot)
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,title,culture_id,culture_display_name,profile_version,profile_hash,locality,setting,stage,status,created_at,updated_at "
                "FROM projects ORDER BY updated_at DESC"
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
            # The deterministic ID may already belong to a retained version that a
            # later prompt superseded. Compiling the exact same dependency again is
            # an explicit request to make that version active; INSERT OR IGNORE
            # alone would otherwise return an invisible, invalidated asset.
            connection.execute(
                """UPDATE assets
                SET status=?, prompt=?, scene_id=COALESCE(?,scene_id), error=NULL, updated_at=?
                WHERE id=? AND status=?""",
                (
                    AssetStatus.pending, prompt, scene_id, now, asset_id,
                    AssetStatus.invalidated,
                ),
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

    def reactivate_asset(self, asset_id: str, status: AssetStatus | str) -> dict[str, Any]:
        """Make one retained version active and invalidate only its sibling versions."""
        asset = self.get_asset(asset_id)
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """UPDATE assets SET status=?, updated_at=?
                WHERE project_id=? AND kind=? AND canonical_id=? AND id<>? AND status<>?""",
                (
                    AssetStatus.invalidated, now, asset["project_id"], asset["kind"],
                    asset["canonical_id"], asset_id, AssetStatus.invalidated,
                ),
            )
            connection.execute(
                "UPDATE assets SET status=?, updated_at=? WHERE id=?",
                (str(status), now, asset_id),
            )
        return self.get_asset(asset_id)

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

    def get_model_run(self, cache_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM model_runs WHERE cache_key=?", (cache_key,)).fetchone()
        return dict(row) if row else None

    def record_model_cache_hit(self, cache_key: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE model_runs SET cache_hit_count=COALESCE(cache_hit_count,0)+1 WHERE cache_key=?",
                (cache_key,),
            )

    def save_model_run(self, **values: Any) -> dict[str, Any]:
        existing = self.get_model_run(values["cache_key"])
        values.setdefault("id", existing["id"] if existing else str(uuid.uuid4()))
        values.setdefault("created_at", utc_now())
        values["output_json"] = canonical_json(values.pop("output")) if "output" in values else None
        for source, target in (
            ("operation_metadata", "operation_metadata_json"), ("input", "input_json"),
            ("input_preview", "input_preview_json"), ("output_preview", "output_preview_json"),
            ("usage", "usage_json"), ("cost", "cost_json"),
        ):
            if source in values:
                values[target] = canonical_json(values.pop(source))
        columns = [
            "id", "project_id", "operation", "provider", "model", "prompt_hash", "input_hash",
            "cache_key", "culture_id", "profile_hash", "brief_revision_id", "output_json", "status", "latency_ms",
            "input_tokens", "output_tokens", "total_tokens", "operation_metadata_json", "input_json",
            "input_preview_json", "output_preview_json", "usage_json", "cost_json", "estimated_cost_usd",
            "pricing_catalog_version", "cost_basis", "cost_calculation_basis", "model_version", "response_id",
            "finish_reason", "attempt_count", "cache_hit_count", "langfuse_trace_id", "langfuse_observation_id",
            "error", "created_at",
        ]
        with self.connect() as connection:
            if existing:
                assignments = ",".join(f"{column}=?" for column in columns if column not in {"id", "created_at"})
                connection.execute(
                    f"UPDATE model_runs SET {assignments} WHERE id=?",
                    [values.get(column) for column in columns if column not in {"id", "created_at"}] + [values["id"]],
                )
            else:
                connection.execute(
                    f"INSERT INTO model_runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    [values.get(column) for column in columns],
                )
        return values

    def save_model_attempt(self, **values: Any) -> dict[str, Any]:
        values.setdefault("id", str(uuid.uuid4()))
        for source, target in (("usage", "usage_json"), ("cost", "cost_json")):
            if source in values:
                values[target] = canonical_json(values.pop(source))
        columns = [
            "id", "model_run_id", "project_id", "invocation_id", "operation", "attempt", "provider", "model",
            "status", "started_at", "completed_at", "latency_ms", "provider_latency_ms", "parse_latency_ms",
            "io_latency_ms", "usage_json", "cost_json", "estimated_cost_usd", "model_version", "response_id", "finish_reason",
            "error_type", "error", "langfuse_trace_id", "langfuse_observation_id",
        ]
        with self.connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO model_call_attempts ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [values.get(column) for column in columns],
            )
        return values

    def list_model_attempts(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_call_attempts WHERE project_id=? ORDER BY started_at,attempt", (project_id,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            for key in ("usage_json", "cost_json"):
                item[key.removesuffix("_json")] = json.loads(item.pop(key)) if item.get(key) else None
            results.append(item)
        return results

    def list_model_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,operation,provider,model,prompt_hash,input_hash,culture_id,profile_hash,brief_revision_id,"
                "status,latency_ms,input_tokens,output_tokens,total_tokens,estimated_cost_usd,pricing_catalog_version,"
                "cost_basis,cost_calculation_basis,model_version,response_id,finish_reason,attempt_count,cache_hit_count,"
                "operation_metadata_json,input_preview_json,output_preview_json,usage_json,cost_json,"
                "langfuse_trace_id,langfuse_observation_id,error,created_at FROM model_runs WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            for key in ("operation_metadata_json", "input_preview_json", "output_preview_json", "usage_json", "cost_json"):
                item[key.removesuffix("_json")] = json.loads(item.pop(key)) if item.get(key) else None
            item["attempts"] = [
                attempt for attempt in self.list_model_attempts(project_id) if attempt["model_run_id"] == item["id"]
            ]
            results.append(item)
        return results

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
