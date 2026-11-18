from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from typing import Any

import httpx
import streamlit as st


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def canonical_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class APIError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int | None = None, unresolved: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.unresolved = unresolved


class Client:
    def __init__(self, base_url: str = BASE_URL, *, pending_store=None):
        self.base_url = base_url.rstrip("/")
        self._local_pending: dict[str, dict[str, Any]] = {}
        self._pending_store = pending_store

    def pending_operations(self) -> dict[str, dict[str, Any]]:
        if self._pending_store is not None:
            return self._pending_store()
        return self._local_pending

    def request(
        self, method: str, path: str, *, json_data: Any = None,
        scope: str | None = None, slot: str | None = None,
    ) -> Any:
        if scope is None:
            return self._send(method, path, json_data=json_data)

        slot = slot or scope
        pending = self.pending_operations()
        operation = pending.get(slot)
        if operation is not None:
            if operation["json_data"] != json_data:
                raise APIError(
                    "A previous submission is unresolved. Check its status before changing the input."
                )
        else:
            operation = {
                "key": str(uuid.uuid4()),
                "scope": scope,
                "method": method,
                "path": path,
                "json_data": deepcopy(json_data),
            }
            pending[slot] = operation

        return self.resend_operation(slot)

    def resend_operation(self, slot: str) -> Any:
        pending = self.pending_operations()
        operation = pending[slot]
        try:
            result = self._send(
                operation["method"],
                operation["path"],
                json_data=operation["json_data"],
                key=operation["key"],
            )
        except APIError as error:
            if not error.unresolved:
                del pending[slot]
            raise

        del pending[slot]
        return result

    def operation_status(self, slot: str) -> dict[str, Any]:
        operation = self.pending_operations()[slot]
        try:
            return self._send(
                "GET",
                f"/v1/operations/{operation['key']}",
                params={"scope": operation["scope"]},
            )
        except APIError as error:
            if error.status_code == 404:
                return {"status": "not_recorded", "outcome": None}
            raise

    def _send(
        self, method: str, path: str, *, json_data: Any = None,
        key: str | None = None, params: dict[str, str] | None = None,
    ) -> Any:
        headers = {"Idempotency-Key": key} if key is not None else {}
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", json=json_data,
                params=params, headers=headers, timeout=300,
            )
        except httpx.RequestError as error:
            message = (
                "The submission outcome is unknown. Check its status before starting another attempt."
                if key is not None else f"Backend unavailable: {error}"
            )
            raise APIError(message, unresolved=key is not None) from error

        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text

            code = detail.get("code") if isinstance(detail, dict) else None
            message = detail.get("message", detail) if isinstance(detail, dict) else detail
            unresolved = key is not None and (
                code == "operation_pending"
                or response.headers.get("Idempotency-Status") == "uncertain"
                or response.status_code >= 500
            )
            raise APIError(
                str(message), status_code=response.status_code, unresolved=unresolved,
            )

        if "application/json" in response.headers.get("content-type", ""):
            try:
                return response.json()
            except ValueError as error:
                raise APIError(
                    "The backend returned an unreadable response.",
                    unresolved=key is not None,
                ) from error

        if key is not None:
            raise APIError(
                "The backend returned an unexpected response. Check the submission status.",
                unresolved=True,
            )
        return response.content

    def health(self): return self.request("GET", "/health")
    def cultures(self): return self.request("GET", "/v1/cultures")
    def culture(self, culture_id: str): return self.request("GET", f"/v1/cultures/{culture_id}")
    def projects(self): return self.request("GET", "/v1/projects")
    def project(self, project_id: str): return self.request("GET", f"/v1/projects/{project_id}")
    def create(self, payload: dict[str, Any]):
        return self.request("POST", "/v1/projects", json_data=payload, scope="create-project")

    def advance(self, project_id: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/advance", scope=f"advance:{project_id}",
        )

    def approve(self, project_id: str, gate: str, revision_id=None, override_reason=None):
        return self.request(
            "POST", f"/v1/projects/{project_id}/approve/{gate}",
            json_data={"revision_id": revision_id, "override_reason": override_reason},
            scope=f"approve:{project_id}:{gate}",
        )

    def patch(self, project_id: str, document_kind: str, field_path: str, value: Any, expected_hash=None):
        return self.request(
            "PATCH", f"/v1/projects/{project_id}/records/{document_kind}/root",
            json_data={
                "document_kind": document_kind, "field_path": field_path,
                "value": value, "expected_hash": expected_hash,
            },
            scope=f"patch:{project_id}:{document_kind}:root",
        )

    def merge(self, project_id: str, record_kind: str, primary_id: str, duplicate_ids: list[str]):
        return self.request(
            "POST", f"/v1/projects/{project_id}/merges",
            json_data={"record_kind": record_kind, "primary_id": primary_id, "duplicate_ids": duplicate_ids},
            scope=f"merge:{project_id}",
        )

    def repair_continuity(self, project_id: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/continuity/repair",
            scope=f"repair-continuity:{project_id}",
        )

    def correct(self, project_id: str, block_id: str, instruction: str, expected_hash: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/corrections",
            json_data={"target_block_id": block_id, "instruction": instruction, "expected_hash": expected_hash},
            scope=f"correct:{project_id}",
        )

    def regenerate_visuals(self, project_id: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/visuals/regenerate",
            scope=f"regenerate-visual-manifest:{project_id}",
        )

    def start_visual_continuity(self, project_id: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/visuals/continuity/start",
            scope=f"start-visual-continuity:{project_id}",
        )

    def compile_scene(self, project_id: str, scene_id: str):
        return self.request(
            "POST", f"/v1/projects/{project_id}/visuals/scenes/{scene_id}/compile",
            scope=f"compile-scene:{project_id}:{scene_id}",
        )

    def generate_asset(self, asset_id: str, retry: bool = False):
        endpoint = "retry" if retry else "generate"
        return self.request(
            "POST", f"/v1/assets/{asset_id}/{endpoint}",
            scope=f"{endpoint}:{asset_id}", slot=f"generation:{asset_id}",
        )

    def correct_asset(self, asset_id: str, instruction: str, style_reference_asset_id=None):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/correct",
            json_data={"instruction": instruction, "style_reference_asset_id": style_reference_asset_id},
            scope=f"correct-asset:{asset_id}",
        )

    def revise_asset_prompt(self, asset_id: str, prompt: str, expected_dependency_hash: str):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/revise-prompt",
            json_data={"prompt": prompt, "expected_dependency_hash": expected_dependency_hash},
            scope=f"revise-asset-prompt:{asset_id}",
        )

    def restore_asset(self, asset_id: str):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/restore", scope=f"restore-asset-version:{asset_id}",
        )

    def approve_asset(self, asset_id: str):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/approve", scope=f"approve-asset:{asset_id}",
        )

    def approve_scene_asset(self, asset_id: str, payload: dict[str, Any]):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/approve-scene",
            json_data=payload, scope=f"approve-scene:{asset_id}",
        )

    def asset_bytes(self, asset_id: str): return self.request("GET", f"/v1/assets/{asset_id}/content")
    def export(self, project_id: str): return self.request("GET", f"/v1/projects/{project_id}/export")


client = Client(
    pending_store=lambda: st.session_state.setdefault("pending_api_operations", {}),
)
