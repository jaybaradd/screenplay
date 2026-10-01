from backend.adaptation_validation import validate_output_script
from backend.cultures.models import ScriptSpec, ScriptValidationPolicy, UnicodeRange
from backend.schemas import AdaptedBlock, AdaptedScene, AdaptedScreenplay, BlockType, CorrectionContent


def screenplay_with(text: str) -> AdaptedScreenplay:
    return AdaptedScreenplay(
        title="Script validation fixture",
        output_script="test_script",
        preservation_summary="Fixture",
        scenes=[AdaptedScene(
            id="scene-1", source_scene_id="scene-1", heading="Scene one", summary="Fixture",
            blocks=[AdaptedBlock(
                id="block-1", source_block_ids=["block-1"], type=BlockType.action,
                adapted_text=text, adaptation_layer_ids=[], cultural_claim_ids=[],
                explanation="Fixture", confidence="high", changed_dimensions=[],
            )],
        )],
    )


def test_profile_driven_script_validation_has_no_named_script_branch():
    script = ScriptSpec(
        id="test_script", display_name="Test Script", font_path="assets/fonts/test.ttf", html_lang="zxx",
        validation=ScriptValidationPolicy(
            unicode_ranges=[UnicodeRange(start="0370", end="03FF")],
            minimum_target_letter_ratio=0.8, minimum_letters_to_check=5,
            enforced_block_types=["action", "dialogue", "transition"],
        ),
    )
    issues = validate_output_script(screenplay_with("This sentence uses another script."), script)
    assert issues[0]["code"] == "output_script_mismatch"
    assert issues[0]["block_id"] == "block-1"
    assert issues[0]["scene_number"] == 1
    assert validate_output_script(screenplay_with("Αυτή είναι μια δοκιμαστική πρόταση."), script) == []


def test_correction_model_output_cannot_control_patch_identity():
    assert set(CorrectionContent.model_fields) == {"replacement_text", "explanation"}
