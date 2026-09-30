from __future__ import annotations

import json
from typing import Any

from backend.cultures.models import CultureRuntimeContext


PROMPT_VERSION = "2026-10-01.culture-profiles-v2-continuity-events"

VISUAL_STYLE_LOCK = (
    "Photorealistic live-action Indian film production reference, real human skin texture and anatomy, "
    "natural camera optics, physically plausible fabric and lighting, restrained documentary realism. "
    "Not an illustration, drawing, painting, animation, anime, comic, vector art, 3D render or game character."
)

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

CONTINUITY EVENT CONTRACT:
- Populate continuity_events using only canonical scene, character, production-element and content-block IDs from your output.
- Add at least one event for every character appearance. Use no_change only after checking that the scene contains no durable
  prop, knowledge, costume, injury, emotion or relationship change for that character.
- Use first_observed_prop when the text reveals that a character already possessed something; do not invent an acquisition.
- Use acquire/release/transfer/derive events for prop movement. A copy is a distinct canonical prop and derive_prop must link
  it to the source prop. Use before_scene only for a source-supported change occurring between two shown scenes.
- Use learn_fact for genuinely acquired knowledge. Do not repeat earlier facts merely to keep them alive; application code
  carries knowledge forward. Use forget_fact only when the source explicitly supports forgetting or retraction.
- Every non-no-change event must cite the source ContentBlock IDs that support it. Do not invent off-screen events merely to
  make the ledger continuous; record uncertainty in the description and confidence.
- Leave state_transitions empty. The application derives before/after snapshots deterministically from continuity_events.

SCENE METADATA COMPLETENESS CONTRACT:
- Evaluate every SceneRecord field for every scene; do not leave metadata null merely because it is outside the slug line.
- Separate a broader location from its exact shootable sub-location whenever the source distinguishes them.
- Infer mood conservatively from observable action, dialogue and atmosphere and phrase it as a production-useful description.
- Capture weather from action as well as headings. Carry persistent weather forward only while the text supports it.
- For sub_location, day_or_date, weather or time that truly is not supplied, write "Not stated in source" instead of null.
- Summaries and dramatic purposes must be scene-specific, not generic boilerplate.
- Populate character roles, relationships, emotional state, entrances and exits when the source supports them. Do not invent them.
- Preserve every source action, dialogue and transition as exactly one ordered ContentBlock. Never summarize, combine or omit beats.
"""


def continuity_repair_prompt(source: dict[str, Any]) -> str:
    return f"""Repair only the screenplay continuity ledger. Do not alter scenes, blocks, characters, production elements or
the StoryContract. Return ContinuityEventExtraction using the exact canonical IDs supplied below.

RULES:
- Cover every character ID listed on every scene with one or more events, or one no_change event after reviewing that appearance.
- Extract durable prop possession/transfer/derivation, knowledge acquisition, costume, injury, emotion and relationship events.
- first_observed_prop means the source first reveals an already-held prop; it is not an acquisition.
- A copied or derived physical object requires its own production-element record. Reuse an existing canonical ID whenever
  possible. If the source explicitly contains a missing prop or costume, propose it in new_production_elements with a temporary
  ID and reference that same temporary ID from events. Do not add decorative or inferred objects.
- before_scene is allowed only where later source text explicitly establishes an intervening change. Explain that inference.
- Every non-no-change event must cite supporting block IDs from the same scene.
- Do not manufacture continuity solely to silence a checker. Prefer an explicit warning when evidence is incomplete.

CANONICAL SOURCE RECORDS:
{json.dumps(source, ensure_ascii=False)}
"""

LAYER_REQUIREMENTS = {
    "verbal": "Use the approved language guide to design authentic dialogue in the exact target variety and writing script: grammar, pronouns, particles, idioms, kinship, honorifics, humour, rhythm, politeness, formality and evidence-backed code-switching. Cover every speaking character and scene. Cite LanguageFeature IDs or forms. Do not invent forms or substitute a related variety.",
    "non_verbal": "Gesture, posture, gaze, touch, silence, greeting, seating, eating and personal space. Condition every proposal on character, relationship and immediate situation.",
    "characters": "Age, class, work, family role, personality, confidence, restraint, affection, anger, relationships and emotional arc. Preserve the source character; prohibit identity assumptions.",
    "visual_world": "Architecture, interiors, wardrobe, fabric, colour, jewellery, grooming, props, food, transport and landscape. Link every visible proposal to locality, period and canonical IDs.",
    "story_world": "Motivations, conflict, stakes, community influence, rituals, humour, resolution, sound and atmosphere. Preserve dramatic function and causal logic.",
    "cultural_precision": "Audit geographic and language-variety specificity, evidence, uncertainty, temporal fit and leakage from unrelated traditions. Produce explicit negative constraints.",
}


def _context(context: CultureRuntimeContext) -> str:
    return json.dumps(context.model_dump(mode="json"), ensure_ascii=False)


def cultural_research_prompt(context: CultureRuntimeContext) -> str:
    if context.source_policy is None:
        raise ValueError("Research context is missing its source policy")
    preferred = ", ".join(context.source_policy.preferred)
    discouraged = ", ".join(context.source_policy.discouraged)
    return f"""Research a narrowly scoped cultural adaptation brief.

TARGET CONTEXT:
{_context(context)}

Research all required dimensions from the profile. Prioritize these source types: {preferred}.
Demote these source types: {discouraged}. Apply this specificity rule: {context.source_policy.locality_specificity_rule}
Never treat a tendency as a universal personality trait. Record time scope, locality scope, uncertainty and traditions that
must not be mixed. Do not invent language forms. Return a concise research narrative with direct citations.

LANGUAGE RESEARCH IS REQUIRED. Find source-backed features of the exact target variety in the selected writing script,
including meaning/function, grammar, pronouns, address and kinship terms, particles, idioms, rhythm/register and realistic
code-switching. Give a direct source URL for every form. Explicitly distinguish the target from every confusable variety
listed in the profile. If evidence is broader than the exact locality or variety, label that limitation.
"""


def cultural_normalize_prompt(context: CultureRuntimeContext, research: str) -> str:
    return f"""Convert the grounded research into the requested CulturalBrief schema.
Use only claims and URLs present in the research or reviewed profile evidence. Omit unsupported detail. Each factual claim
must have a source URL, origin, scope, period, confidence, applicable layers, uncertainty and prohibited extrapolations.
Use origin=grounded_research for researched claims and origin=reviewed_profile_evidence only for supplied reviewed evidence.

Copy these immutable identity fields exactly from TARGET CONTEXT: culture_id, profile_version, profile_hash, display_name as
culture, locality, setting, period, and output_script. Build language_guide only from source-backed forms. Its target_variety
and writing_script must exactly match the target context. Every LanguageFeature requires a form, function, usage constraint,
confidence and direct source URL. If evidence is insufficient, keep the guide sparse and record the gap; never invent examples.

Return structured constraints with unique IDs and classify each as language_boundary, visual_boundary, research_boundary or
identity_boundary. Profile policies must use origin=profile_policy and retain their IDs. A grounded constraint must use
origin=grounded_research and cite supporting claim IDs. Use only the six approved layer names.

TARGET CONTEXT:
{_context(context)}

GROUNDED RESEARCH:
{research}
"""


def layer_prompt(
    layer: str, source: dict[str, Any], context: CultureRuntimeContext,
) -> str:
    return f"""You are the {layer} planning specialist for a culturally precise screenplay adaptation.
{PRESERVATION_RULES}

TARGET CULTURAL CONTEXT:
{_context(context)}

YOUR ONLY LAYER:
{LAYER_REQUIREMENTS[layer]}

Return one LayerPlan for layer={layer}. Every decision must include source observations, proposed changes, affected stable IDs,
preserved invariants, supporting CulturalBrief claim IDs, uncertainty, risks, do-not-change instructions and negative constraints.

SOURCE RECORDS:
{json.dumps(source, ensure_ascii=False)}
"""


def synthesis_prompt(layers: list[dict[str, Any]], story_contract: dict[str, Any]) -> str:
    return f"""Synthesize the six specialist plans into one AdaptationPlan without dropping or rewriting their traceability.
{PRESERVATION_RULES}

Identify conflicts between verbal, non-verbal, character, visual-world, story-world and cultural-precision proposals.
Resolve a conflict only when approved evidence and the StoryContract make the resolution unambiguous. Any proposal that
changes central dramatic purpose, relationship invariants, causal plot logic or emotional arc must become an explicit
deeper_change_request and must not be silently approved. Preserve all six LayerPlan objects in the output.

STORY CONTRACT:
{json.dumps(story_contract, ensure_ascii=False)}

SIX LAYER PLANS:
{json.dumps(layers, ensure_ascii=False)}
"""


def adaptation_prompt(
    scene: dict[str, Any], source: dict[str, Any], brief: dict[str, Any], plan: dict[str, Any],
    context: CultureRuntimeContext,
) -> str:
    if context.language_policy is None:
        raise ValueError("Scene-adaptation context is missing its language policy")
    fallback = context.language_policy.fallback_language or "No fallback language"
    confusable = ", ".join(context.confusable_languages) or "none declared"
    return f"""Adapt exactly one screenplay scene for the approved target cultural context.
{PRESERVATION_RULES}

TARGET CULTURAL CONTEXT:
{_context(context)}

LANGUAGE CONTRACT:
- Dialogue target: {context.target_variety}, written in {context.output_script_name}.
- Writing script alone is not evidence that dialogue belongs to the target variety.
- Fallback language ({fallback}) may appear only when an approved register or code-switching rule justifies it.
- Never substitute these confusable varieties: {confusable}.
- Use only the approved language guide and verbal-plan decisions. Mark uncertainty in explanations and confidence.
- Screen directions may use a clear production register appropriate to the selected script; dialogue must follow the target variety.

LOSSLESS BLOCK CONTRACT:
- Return exactly {len(scene.get('blocks', []))} blocks, one output block for each source block, in identical order.
- Copy each source block id as both the adapted block id and the sole item in source_block_ids.
- Preserve block type and speaker_id. Do not merge, split, add, summarize or omit blocks.

Use only approved plan decisions and cultural claims relevant to this scene. Adapt spoken and non-verbal behaviour, character
context, visible production world and atmosphere without turning the scene into cultural exhibition. Explain each important change.

SCENE:
{json.dumps(scene, ensure_ascii=False)}

SOURCE CONTRACT AND CANONICAL RECORDS:
{json.dumps(source, ensure_ascii=False)}

APPROVED CULTURAL BRIEF:
{json.dumps(brief, ensure_ascii=False)}

APPROVED PLAN:
{json.dumps(plan, ensure_ascii=False)}
"""


def adaptation_repair_prompt(
    scene: dict[str, Any], source: dict[str, Any], brief: dict[str, Any], plan: dict[str, Any],
    context: CultureRuntimeContext, invalid_output: dict[str, Any], errors: list[str],
) -> str:
    return f"""Repair a structurally incomplete screenplay adaptation. Return the COMPLETE AdaptedScene, not a patch.
The previous output failed these deterministic checks:
{json.dumps(errors, ensure_ascii=False)}

Follow every language and lossless-block contract below. Account for every source block exactly once.

{adaptation_prompt(scene, source, brief, plan, context)}

INVALID PREVIOUS OUTPUT:
{json.dumps(invalid_output, ensure_ascii=False)}
"""


def language_audit_prompt(
    adapted: dict[str, Any], brief: dict[str, Any], context: CultureRuntimeContext,
) -> str:
    if context.language_policy is None:
        raise ValueError("Language-audit context is missing its language policy")
    fallback = context.language_policy.fallback_language or "the configured fallback language"
    confusable = ", ".join(context.confusable_languages) or "related or invented varieties"
    return f"""Audit dialogue language only. Determine whether each dialogue block is genuinely consistent with the approved
target variety and language guide, rather than merely using the selected writing script.

TARGET CULTURAL CONTEXT:
{_context(context)}

Use exact block IDs. Put supported dialogue in supported_target_variety_block_ids. Put unsupported {fallback} dialogue in
unsupported_fallback_language_block_ids. Put {confusable}, invented or guide-inconsistent forms in
mixed_or_unapproved_variety_block_ids. Do not penalize directions, headings, names or explicitly approved code-switching.
A scene passes only when its dialogue is supported or its code-switching is justified. Be conservative and evidence-bound.

APPROVED CULTURAL BRIEF:
{json.dumps(brief, ensure_ascii=False)}

ADAPTED SCREENPLAY:
{json.dumps(adapted, ensure_ascii=False)}
"""


def visual_manifest_prompt(
    source: dict[str, Any], adapted: dict[str, Any], plan: dict[str, Any],
    context: CultureRuntimeContext, appearance_scaffold: list[dict[str, Any]], scene_scaffold: list[dict[str, Any]],
) -> str:
    return f"""Create a production-ready VisualManifest for the approved cultural adaptation.

TARGET VISUAL CULTURAL CONTEXT:
{_context(context)}

This is visual production planning, not dialogue generation. Use only visually relevant approved cultural evidence and
constraints. Do not copy linguistic grammar, vocabulary or writing-system constraints into image prompts.

CHARACTER/APPEARANCE REQUIREMENTS:
- Return exactly one AppearanceSpec for every scaffold, preserving supplied IDs and traceability fields.
- Make identity_description specific: apparent age, face shape, complexion without colourism, physique, posture, expression,
  occupational wear and stable identity anchors. Do not infer caste, religion or wealth beyond approved evidence.
- Make costume_description character-, occupation-, weather- and class-context specific: garment pieces, practical drape,
  fabric, colour, wear, footwear, jewellery/accessories and continuity. Avoid decorative festival styling unless required.
- Make grooming_description specific and stable.
- The final prompt must include identity, costume and grooming and request a neutral close portrait plus full-body wardrobe view.
- Prompts must be meaningfully different between characters; changing only a name is invalid.
- Every prompt must preserve this exact project-wide style: {VISUAL_STYLE_LOCK}

SCENE REQUIREMENTS:
- Return exactly one SceneVisualSpec for every scaffold and preserve all supplied IDs and traceability fields.
- Build prompts from the adapted scene, approved plan, current appearances, location, props, action, composition, time,
  weather, lighting, mood and atmosphere.
- Describe one precise cinematic moment. Do not put dialogue, captions, subtitles or screenplay text inside the image.
- negative_prompt must contain visual constraints only.
- Every scene must use the same photorealistic live-action visual language as the character sheets.

APPEARANCE SCAFFOLD:
{json.dumps(appearance_scaffold, ensure_ascii=False)}

SCENE SCAFFOLD:
{json.dumps(scene_scaffold, ensure_ascii=False)}

SOURCE CANONICAL RECORDS:
{json.dumps(source, ensure_ascii=False)}

APPROVED ADAPTED SCREENPLAY:
{json.dumps(adapted, ensure_ascii=False)}

APPROVED RELEVANT PLAN:
{json.dumps(plan, ensure_ascii=False)}
"""


def correction_prompt(
    target: dict[str, Any], instruction: str, context: dict[str, Any], culture: CultureRuntimeContext,
) -> str:
    return f"""Return one surgical CorrectionPatch for the selected target only.
Do not rewrite any other block. Keep target_id and precondition_hash exactly as provided.
{PRESERVATION_RULES}

TARGET CULTURAL CONTEXT:
{_context(culture)}

TARGET:
{json.dumps(target, ensure_ascii=False)}

USER INSTRUCTION:
{instruction}

NEIGHBOUR CONTEXT AND CONSTRAINTS:
{json.dumps(context, ensure_ascii=False)}
"""
