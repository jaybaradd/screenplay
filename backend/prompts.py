from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "2026-09-29.v1"

PRESERVATION_RULES = """
PRESERVE THE STORY:
- Keep the central dramatic purpose, causal plot logic, character relationships and emotional arc recognizable.
- Never add a cultural detail that reverses motivation, creates a plot hole, or assigns caste, religion, class or values without source evidence.
- If a deeper change is genuinely necessary, return it as an explicit request with its reason; do not silently apply it.
- Retain stable scene, character, entity and block IDs supplied in the input.
"""

EXTRACTION_PROMPT = """You are a screenplay extraction engine. Parse the complete source into the supplied schema.
Preserve scene order and every material action/dialogue beat. Use stable IDs already supplied in the scaffold when present.
Do not culturally adapt anything. Do not resolve ambiguous aliases silently: preserve aliases and add a warning.
Capture production elements and before/event/after continuity state only when supported by text.
Create a StoryContract describing dramatic purpose, relationships, plot invariants and emotional arc.
"""

CULTURAL_RESEARCH_PROMPT = """Research a narrowly scoped contemporary cultural adaptation brief for:
culture={culture}; locality={locality}; setting={setting}; period={period}; output script={output_script}.

Prioritize institutional, academic, linguistic, museum, government, or locality-specific sources. Demote tourism marketing,
unsourced listicles and generic Rajasthan material. Separately investigate verbal, non-verbal, character/social context,
visual world, story world, and cultural precision. Never treat a tendency as a universal personality trait. Record uncertainty,
time scope, locality scope, and traditions that must not be mixed. Do not invent dialect phrases. Return a concise research
narrative with citations; it will be normalized in a second call.
"""

CULTURAL_NORMALIZE_PROMPT = """Convert the grounded research below into the requested CulturalBrief schema.
Use only claims and URLs present in the supplied research. Omit unsupported detail. Each claim must have a source URL,
scope, period, confidence, applicable layers, uncertainty, and prohibited extrapolations. Explicitly block generic palace,
camel, sand-dune, wedding, tourist-folk and unrelated Rajasthani styling unless demanded by the screenplay.

GROUNDED RESEARCH:
{research}
"""

LAYER_REQUIREMENTS = {
    "verbal": "Dialect/register, idioms, kinship, honorifics, humour, rhythm, politeness, formality and code-switching. Tie each proposal to speaker, relationship and scene. Do not invent dialect forms.",
    "non_verbal": "Gesture, posture, gaze, touch, silence, greeting, seating, eating and personal space. Condition every proposal on character, relationship and immediate situation.",
    "characters": "Age, class, work, family role, personality, confidence, restraint, affection, anger, relationships and emotional arc. Preserve the source character; prohibit identity assumptions.",
    "visual_world": "Architecture, interiors, wardrobe, fabric, colour, jewellery, grooming, props, food, transport and landscape. Link every visible proposal to locality, period and canonical IDs.",
    "story_world": "Motivations, conflict, stakes, community influence, rituals, humour, resolution, sound and atmosphere. Preserve dramatic function and causal logic.",
    "cultural_precision": "Audit geographic/dialect specificity, evidence, uncertainty, temporal fit and leakage from unrelated traditions. Produce explicit negative constraints.",
}


def layer_prompt(layer: str, source: dict[str, Any], brief: dict[str, Any]) -> str:
    return f"""You are the {layer} planning specialist for a culturally precise screenplay adaptation.
{PRESERVATION_RULES}

YOUR ONLY LAYER:
{LAYER_REQUIREMENTS[layer]}

Return one LayerPlan for layer={layer}. Every decision must include source observations, proposed changes, affected stable IDs,
preserved invariants, supporting CulturalBrief claim IDs, uncertainty, risks, do-not-change instructions and negative constraints.

SOURCE RECORDS:
{json.dumps(source, ensure_ascii=False)}

APPROVED CULTURAL BRIEF:
{json.dumps(brief, ensure_ascii=False)}
"""


def synthesis_prompt(layers: list[dict[str, Any]], story_contract: dict[str, Any]) -> str:
    return f"""Synthesize the six specialist plans into one AdaptationPlan without dropping or rewriting their traceability.
{PRESERVATION_RULES}

Identify conflicts between verbal, non-verbal, character, visual-world, story-world and cultural-precision proposals.
Resolve a conflict only when the CulturalBrief evidence and StoryContract make the resolution unambiguous. Any proposal that
changes central dramatic purpose, relationship invariants, causal plot logic or emotional arc must become an explicit
deeper_change_request and must not be silently approved. Preserve all six LayerPlan objects in the output.

STORY CONTRACT:
{json.dumps(story_contract, ensure_ascii=False)}

SIX LAYER PLANS:
{json.dumps(layers, ensure_ascii=False)}
"""


def adaptation_prompt(scene: dict[str, Any], source: dict[str, Any], brief: dict[str, Any], plan: dict[str, Any]) -> str:
    return f"""Adapt exactly one screenplay scene into Devanagari for Maidani Mewari cultural context.
{PRESERVATION_RULES}

Use only approved plan decisions and cultural claims relevant to this scene. If exact dialect wording is uncertain, prefer
natural restrained Hindi/Devanagari and mark low confidence rather than fabricate. Adapt spoken and non-verbal behaviour,
character context, visible production world and atmosphere without turning the scene into cultural exhibition.
Map every adapted block to source block IDs and explain each important change.

SCENE:
{json.dumps(scene, ensure_ascii=False)}

SOURCE CONTRACT AND CANONICAL RECORDS:
{json.dumps(source, ensure_ascii=False)}

CULTURAL BRIEF:
{json.dumps(brief, ensure_ascii=False)}

APPROVED PLAN:
{json.dumps(plan, ensure_ascii=False)}
"""


def correction_prompt(target: dict[str, Any], instruction: str, context: dict[str, Any]) -> str:
    return f"""Return one surgical CorrectionPatch for the selected target only.
Do not rewrite any other block. Keep target_id and precondition_hash exactly as provided.
{PRESERVATION_RULES}

TARGET:
{json.dumps(target, ensure_ascii=False)}

USER INSTRUCTION:
{instruction}

NEIGHBOUR CONTEXT AND CONSTRAINTS:
{json.dumps(context, ensure_ascii=False)}
"""
