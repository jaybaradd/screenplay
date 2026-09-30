from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Any

import httpx


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def canonical_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class APIError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, *, json_data: Any = None, idempotent: bool = False) -> Any:
        headers = {"Idempotency-Key": str(uuid.uuid4())} if idempotent else {}
        try:
            response = httpx.request(method, f"{self.base_url}{path}", json=json_data, headers=headers, timeout=300)
        except httpx.HTTPError as error:
            raise APIError(f"Backend unavailable: {error}") from error
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise APIError(str(detail))
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.content

    def health(self): return self.request("GET", "/health")
    def cultures(self): return self.request("GET", "/v1/cultures")
    def culture(self, culture_id: str): return self.request("GET", f"/v1/cultures/{culture_id}")
    def projects(self): return self.request("GET", "/v1/projects")
    def project(self, project_id: str): return self.request("GET", f"/v1/projects/{project_id}")
    def create(self, payload: dict[str, Any]): return self.request("POST", "/v1/projects", json_data=payload, idempotent=True)
    def advance(self, project_id: str): return self.request("POST", f"/v1/projects/{project_id}/advance", idempotent=True)
    def approve(self, project_id: str, gate: str, revision_id: str | None = None, override_reason: str | None = None):
        return self.request("POST", f"/v1/projects/{project_id}/approve/{gate}", json_data={"revision_id": revision_id, "override_reason": override_reason}, idempotent=True)
    def patch(self, project_id: str, document_kind: str, field_path: str, value: Any, expected_hash: str | None = None):
        return self.request("PATCH", f"/v1/projects/{project_id}/records/{document_kind}/root", json_data={"document_kind": document_kind, "field_path": field_path, "value": value, "expected_hash": expected_hash}, idempotent=True)
    def merge(self, project_id: str, record_kind: str, primary_id: str, duplicate_ids: list[str]):
        return self.request("POST", f"/v1/projects/{project_id}/merges", json_data={"record_kind": record_kind, "primary_id": primary_id, "duplicate_ids": duplicate_ids}, idempotent=True)
    def repair_continuity(self, project_id: str):
        return self.request("POST", f"/v1/projects/{project_id}/continuity/repair", idempotent=True)
    def correct(self, project_id: str, block_id: str, instruction: str, expected_hash: str):
        return self.request("POST", f"/v1/projects/{project_id}/corrections", json_data={"target_block_id": block_id, "instruction": instruction, "expected_hash": expected_hash}, idempotent=True)
    def regenerate_visuals(self, project_id: str):
        return self.request("POST", f"/v1/projects/{project_id}/visuals/regenerate", idempotent=True)
    def start_visual_continuity(self, project_id: str):
        return self.request("POST", f"/v1/projects/{project_id}/visuals/continuity/start", idempotent=True)
    def compile_scene(self, project_id: str, scene_id: str):
        return self.request("POST", f"/v1/projects/{project_id}/visuals/scenes/{scene_id}/compile", idempotent=True)
    def generate_asset(self, asset_id: str, retry: bool = False):
        endpoint = "retry" if retry else "generate"
        return self.request("POST", f"/v1/assets/{asset_id}/{endpoint}", idempotent=True)
    def correct_asset(self, asset_id: str, instruction: str, style_reference_asset_id: str | None = None):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/correct",
            json_data={"instruction": instruction, "style_reference_asset_id": style_reference_asset_id},
            idempotent=True,
        )
    def revise_asset_prompt(self, asset_id: str, prompt: str, expected_dependency_hash: str):
        return self.request(
            "POST", f"/v1/assets/{asset_id}/revise-prompt",
            json_data={"prompt": prompt, "expected_dependency_hash": expected_dependency_hash},
            idempotent=True,
        )
    def restore_asset(self, asset_id: str):
        return self.request("POST", f"/v1/assets/{asset_id}/restore", idempotent=True)
    def approve_asset(self, asset_id: str): return self.request("POST", f"/v1/assets/{asset_id}/approve", idempotent=True)
    def approve_scene_asset(self, asset_id: str, payload: dict[str, Any]):
        return self.request("POST", f"/v1/assets/{asset_id}/approve-scene", json_data=payload, idempotent=True)
    def asset_bytes(self, asset_id: str): return self.request("GET", f"/v1/assets/{asset_id}/content")
    def export(self, project_id: str): return self.request("GET", f"/v1/projects/{project_id}/export")


client = Client()
