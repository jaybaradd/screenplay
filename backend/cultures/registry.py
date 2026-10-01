from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.cultures.models import (
    CultureProfile, CultureProfileSnapshot, CultureProfileSummary, ReviewedEvidence, TargetSelection,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_ROOT = Path(__file__).resolve().parent / "profiles"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CultureRegistry:
    def __init__(self, profile_root: Path = DEFAULT_PROFILE_ROOT):
        self.profile_root = profile_root
        self._snapshots: dict[str, CultureProfileSnapshot] = {}
        self.reload()

    def reload(self) -> None:
        snapshots: dict[str, CultureProfileSnapshot] = {}
        if not self.profile_root.exists():
            raise RuntimeError(f"Culture profile directory does not exist: {self.profile_root}")
        for profile_path in sorted(self.profile_root.glob("*/profile.json")):
            profile = CultureProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
            if profile.culture_id in snapshots:
                raise RuntimeError(f"Duplicate culture profile id: {profile.culture_id}")
            evidence_path = profile_path.with_name("reviewed_evidence.json")
            evidence = ReviewedEvidence.model_validate_json(
                evidence_path.read_text(encoding="utf-8") if evidence_path.exists() else "{}"
            )
            for script in profile.supported_scripts:
                font = (ROOT / script.font_path).resolve()
                if ROOT not in font.parents or not font.is_file():
                    raise RuntimeError(
                        f"Culture profile {profile.culture_id} references a missing or unsafe font: {script.font_path}"
                    )
                if profile.production_enabled and script.validation is None:
                    raise RuntimeError(
                        f"Production culture profile {profile.culture_id} has no validation policy for script {script.id}"
                    )
            payload = {
                "profile": profile.model_dump(mode="json"),
                "reviewed_evidence": evidence.model_dump(mode="json"),
            }
            snapshots[profile.culture_id] = CultureProfileSnapshot(
                profile=profile,
                reviewed_evidence=evidence,
                profile_hash=hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest(),
            )
        if not snapshots:
            raise RuntimeError("No valid culture profiles were found")
        self._snapshots = snapshots

    def snapshot(self, culture_id: str) -> CultureProfileSnapshot:
        try:
            return self._snapshots[culture_id].model_copy(deep=True)
        except KeyError as error:
            raise ValueError(f"Unknown culture profile: {culture_id}") from error

    def summaries(self, *, production_only: bool = True) -> list[CultureProfileSummary]:
        profiles = [item.profile for item in self._snapshots.values()]
        if production_only:
            profiles = [item for item in profiles if item.production_enabled]
        return [CultureProfileSummary(
            culture_id=item.culture_id,
            version=item.version,
            display_name=item.display_name,
            language_name=item.language_name,
            target_variety=item.target_variety,
            geographic_scope=item.geographic_scope,
            limitations=item.limitations,
            supported_scripts=item.supported_scripts,
            supported_settings=item.supported_settings,
            supported_periods=item.supported_periods,
        ) for item in profiles]

    def validate_selection(self, selection: TargetSelection) -> CultureProfileSnapshot:
        snapshot = self.snapshot(selection.culture_id)
        profile = snapshot.profile
        if not selection.locality.strip():
            raise ValueError("Exact locality is required")
        if selection.setting not in profile.supported_settings:
            raise ValueError(f"Unsupported setting for {profile.display_name}: {selection.setting}")
        if selection.period not in {item.id for item in profile.supported_periods}:
            raise ValueError(f"Unsupported period for {profile.display_name}: {selection.period}")
        if selection.output_script not in {item.id for item in profile.supported_scripts}:
            raise ValueError(f"Unsupported output script for {profile.display_name}: {selection.output_script}")
        return snapshot


culture_registry = CultureRegistry()
