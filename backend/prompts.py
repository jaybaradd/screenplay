from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "2026-09-30.v2"

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

SCENE METADATA COMPLETENESS CONTRACT:
- Evaluate every SceneRecord field for every scene; do not leave metadata null merely because it is outside the slug line.
- Split a compound place such as "FARMHOUSE DINING ROOM" into location="FARMHOUSE" and sub_location="DINING ROOM".
- Infer mood conservatively from observable action, dialogue and atmosphere and phrase it as a production-useful description.
- Capture weather from action as well as headings. Carry persistent weather forward only while the text supports it.
- For sub_location, day_or_date, weather or time that truly is not supplied, write "Not stated in source" instead of null.
- Summaries and dramatic purposes must be scene-specific, not generic boilerplate.
- Populate character roles, relationships, emotional state, entrances and exits when the source supports them. Do not invent them.
- Preserve every source action, dialogue and transition as exactly one ordered ContentBlock. Never summarize, combine or omit beats.
"""

CULTURAL_RESEARCH_PROMPT = """Research a narrowly scoped contemporary cultural adaptation brief for:
culture={culture}; locality={locality}; setting={setting}; period={period}; output script={output_script}.

Prioritize institutional, academic, linguistic, museum, government, or locality-specific sources. Demote tourism marketing,
unsourced listicles and generic Rajasthan material. Separately investigate verbal, non-verbal, character/social context,
visual world, story world, and cultural precision. Never treat a tendency as a universal personality trait. Record uncertainty,
time scope, locality scope, and traditions that must not be mixed. Do not invent dialect phrases. Return a concise research
narrative with citations; it will be normalized in a second call.

LANGUAGE RESEARCH IS A REQUIRED SEPARATE SECTION. Find source-backed features of the exact target variety, including usable
Devanagari forms, meaning/function, grammar, pronouns, address and kinship terms, particles, idioms, rhythm/register and realistic
Hindi code-switching. Give a source URL for every form. Prefer dictionaries, grammars, linguistic surveys, language archives,
academic work and locally authored language material. Do not substitute Marwari, generic Rajasthani or standard Hindi for
Maidani Mewari. If a source covers broader Mewari rather than the exact locality, label that limitation explicitly.
"""

CULTURAL_NORMALIZE_PROMPT = """Convert the grounded research below into the requested CulturalBrief schema.
Use only claims and URLs present in the supplied research. Omit unsupported detail. Each claim must have a source URL,
scope, period, confidence, applicable layers, uncertainty, and prohibited extrapolations. Explicitly block generic palace,
camel, sand-dune, wedding, tourist-folk and unrelated Rajasthani styling unless demanded by the screenplay.

Build dialect_guide only from source-backed language material in the research. Its target_variety must be "Maidani Mewari"
and writing_script must be "Devanagari". Every DialectFeature needs an exact Devanagari form, function, usage constraint,
confidence and direct source URL. Use only these layer names in claims: verbal, non_verbal, characters, visual_world,
story_world, cultural_precision. Never relabel Marwari or standard Hindi material as Mewari. If evidence is insufficient,
leave features sparse and record the gap in open_questions rather than inventing examples.

GROUNDED RESEARCH:
{research}
"""

LAYER_REQUIREMENTS = {
    "verbal": "Use the approved dialect_guide to design actual Maidani Mewari dialogue in Devanagari: grammar, pronouns, particles, idioms, kinship, honorifics, humour, rhythm, politeness, formality and code-switching. Cover every speaking character and scene. Cite DialectFeature IDs/forms in proposed changes. Standard Hindi is not the target language, and Marwari is not a substitute. Do not invent dialect forms.",
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

LANGUAGE CONTRACT:
- Dialogue target: Maidani Mewari written in Devanagari. Devanagari is a script, not evidence of Mewari.
- Standard Hindi may appear only when the approved character/register or code-switching rules justify it. It is not a fallback.
- Never substitute Marwari, generic Rajasthani or invented "Rajasthani-sounding" forms.
- Use the approved dialect_guide and verbal-plan decisions. Mark uncertainty in explanations and confidence.
- Screen directions and headings may use clear Hindi in Devanagari, while dialogue must follow the target variety.

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

CULTURAL BRIEF:
{json.dumps(brief, ensure_ascii=False)}

APPROVED PLAN:
{json.dumps(plan, ensure_ascii=False)}
"""


def adaptation_repair_prompt(
    scene: dict[str, Any], source: dict[str, Any], brief: dict[str, Any], plan: dict[str, Any],
    invalid_output: dict[str, Any], errors: list[str],
) -> str:
    return f"""Repair a structurally incomplete screenplay adaptation. Return the COMPLETE AdaptedScene, not a patch.
The previous output failed these deterministic checks:
{json.dumps(errors, ensure_ascii=False)}

Follow every LANGUAGE CONTRACT and LOSSLESS BLOCK CONTRACT below. Account for every source block exactly once.

{adaptation_prompt(scene, source, brief, plan)}

INVALID PREVIOUS OUTPUT (use only to understand the failure; do not copy its omissions):
{json.dumps(invalid_output, ensure_ascii=False)}
"""


def dialect_audit_prompt(adapted: dict[str, Any], brief: dict[str, Any]) -> str:
    return f"""Audit dialogue language only. Determine whether each dialogue block is genuinely consistent with the approved
Maidani Mewari dialect guide in Devanagari, rather than standard Hindi merely written in Devanagari.

Use exact block IDs. Put unsupported standard-Hindi dialogue in generic_hindi_block_ids. Put apparent Marwari, generic
Rajasthani, invented, or guide-inconsistent forms in unapproved_or_mixed_block_ids. Do not penalize screenplay directions,
headings, character names, or explicitly approved situational Hindi code-switching. A scene passes only when its dialogue is
supported or its code-switching is justified. Be conservative and evidence-bound.

APPROVED CULTURAL BRIEF AND DIALECT GUIDE:
{json.dumps(brief, ensure_ascii=False)}

ADAPTED SCREENPLAY:
{json.dumps(adapted, ensure_ascii=False)}
"""


def visual_manifest_prompt(
    source: dict[str, Any], adapted: dict[str, Any], plan: dict[str, Any], brief: dict[str, Any],
    appearance_scaffold: list[dict[str, Any]], scene_scaffold: list[dict[str, Any]],
) -> str:
    return f"""Create a production-ready VisualManifest for a contemporary Maidani Mewari screenplay adaptation.

This is visual production planning, not dialogue generation. Use only visually relevant cultural evidence and negative
constraints. Do not copy linguistic grammar, vocabulary, or script constraints into image prompts.

CHARACTER/APPEARANCE REQUIREMENTS:
- Return exactly one AppearanceSpec for every appearance scaffold, preserving every supplied id, character_id and scene_ids.
- Make identity_description specific: apparent age, face shape, complexion without colourism, physique, posture, expression,
  occupational wear and stable identity anchors. Do not infer caste, religion, or wealth beyond approved evidence.
- Make costume_description character-, occupation-, weather- and class-context specific: garment pieces, practical drape,
  fabric, colour, wear, footwear, jewellery/accessories and continuity. Avoid decorative festival styling unless required.
- Make grooming_description specific and stable: hair, facial hair where relevant, grooming and practical weather effects.
- The final prompt must explicitly include the identity, costume and grooming descriptions and request a two-panel sheet:
  neutral close portrait plus full-body wardrobe reference, plain production-reference background, no written labels.
- Prompts must be meaningfully different between characters; changing only a name is invalid.
- Every prompt must preserve this exact project-wide style: {VISUAL_STYLE_LOCK}

SCENE REQUIREMENTS:
- Return exactly one SceneVisualSpec for every scene scaffold and preserve supplied ids, scene_id, appearance_ids,
  location_id and prop_ids.
- Build prompts from the ADAPTED scene, approved visual/story/non-verbal plan, current appearances, location, props,
  action, composition, time, weather, lighting, mood and atmosphere.
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

APPROVED VISUAL CULTURAL EVIDENCE:
{json.dumps(brief, ensure_ascii=False)}
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
