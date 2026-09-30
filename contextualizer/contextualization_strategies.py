# contextualization_strategies.py
"""Contextualization strategies for train-time self-distillation.

The module implements the semantic policies registered in ``STRATEGY_REGISTRY``.
Strategies operate on dataset-adapter abstractions and return chat messages, not
rendered tokenizer strings. Static strategies can be precomputed during dataset
mapping; dynamic strategies declare whether they require a student response,
auxiliary feedback/rationale generation, or sibling-response selection so the
trainer can build the final teacher prompt at runtime.

Prompt text is defined separately in ``contextualization_prompt_templates.py``.
Dataset-specific schema handling remains in ``dataset_adapters.py``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .contextualization_strategies_base import (
    FEEDBACK_MAX_USE_DATASET_DEFAULT,
    _adapter_question_text,
    _adapter_compact_answer_text,
    _adapter_full_reference_text,
    _adapter_full_reference_text_old,
    _adapter_short_hint_text,
    ContextStrategy,
    TwoStageStrategy
)
from .contextualization_prompt_templates import (
    CONCRETE_SELF_FEEDBACK_STAGE1_TEMPLATES,
    GOLD_ANSWER_AS_EXAMPLE_PROMPT_TEMPLATES,
    GOLD_ANSWER_AS_EXAMPLE2_PROMPT_TEMPLATES,
    GOLD_ANSWER_AS_GUIDANCE_PROMPT_TEMPLATES,
    GOLD_HINT_V2_PROMPT_TEMPLATES,
    GOLD_HINT_V3_PROMPT_TEMPLATES,
    GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES,
    GOLD_RESPONSE_AS_GUIDANCE_PROMPT_TEMPLATES,
    GOLD_RESPONSE_INV_PROMPT_TEMPLATES,
    NO_CONTEXT_PROMPT_TEMPLATES,
    RATIONALIZATION_PROMPT_TEMPLATES,
    RATIONALIZATION_STAGE1_TEMPLATES,
    SELF_FEEDBACK_PROMPT_TEMPLATES,
    SELF_FEEDBACK_STAGE1_TEMPLATES,
    STATIC_REASONING_PROMPT_TEMPLATES,
    STRUCTURED_SELF_FEEDBACK_PROMPT_TEMPLATES,
    STRUCTURED_SELF_FEEDBACK_STAGE1_TEMPLATES,
    GOLD_RESPONSE_REWRITE_AS_EXAMPLE_PROMPT_TEMPLATES,
    GOLD_RESPONSE_REWRITE_STAGE1_TEMPLATES,
    GOLD_RESPONSE_REWRITE_AS_EXAMPLE_V2_PROMPT_TEMPLATES,
    GOLD_RESPONSE_REWRITE_V2_STAGE1_TEMPLATES,
    SDPO_SIBLING_FEEDBACK_PROMPT_TEMPLATES,
)

# =============================================================================
# Generic rendering helpers
# =============================================================================


def render_reference(value: Any) -> str:
    """
    Convert arbitrary references/golden answers into readable text.

    This mirrors analyze_contextualization.render_reference and is intentionally
    conservative:
      - strings are returned unchanged;
      - JSON-serializable objects are pretty-printed;
      - everything else falls back to str(...).
    """
    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)




# =============================================================================
# ToolUse-specific helper functions
# =============================================================================


@dataclass
class TooluseFailureInfo:
    failure_mode: str
    pred_actions: list[str]
    gt_actions: list[str]
    pred_inputs: dict[str, Any]
    gt_inputs: dict[str, Any]


def classify_tooluse_failure(adapter, response: str, reference: Any) -> TooluseFailureInfo:
    """
    Classify ToolUse errors using the same extraction logic as ToolUseAdapter.

    Failure modes:
      - correct
      - missing_action
      - wrong_actions
      - wrong_inputs
      - malformed_or_missing_inputs
      - other
    """
    pred_actions = adapter.extract_actions(response)
    pred_inputs = adapter.extract_action_inputs(response)

    gt_actions: list[str] = []
    gt_inputs: dict[str, Any] = {}

    if isinstance(reference, list):
        for item in reference:
            if not isinstance(item, dict):
                continue

            if "Action" in item:
                gt_actions.append(item["Action"])

            if "Action_Input" in item:
                try:
                    gt_inputs.update(json.loads(item["Action_Input"]))
                except Exception:
                    pass

    actions_match = Counter(pred_actions) == Counter(gt_actions)
    inputs_match = pred_inputs == gt_inputs

    if actions_match and inputs_match:
        mode = "correct"
    elif not pred_actions:
        mode = "missing_action"
    elif not pred_inputs and gt_inputs:
        mode = "malformed_or_missing_inputs"
    elif not actions_match:
        mode = "wrong_actions"
    elif not inputs_match:
        mode = "wrong_inputs"
    else:
        mode = "other"

    return TooluseFailureInfo(
        failure_mode=mode,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
        pred_inputs=pred_inputs,
        gt_inputs=gt_inputs,
    )


# =============================================================================
# Static / precomputable strategies
# =============================================================================

class LegacyFullReferenceMixin:
    """
    Mixin for strategies that intentionally use the compatibility $full_reference rendering.

    This lets compatibility strategies reuse the shared prompt templates while
    changing only the value substituted into ``$full_reference``.
    """

    def build_template_variables(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
    ) -> dict[str, Any]:
        variables = super().build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
        )
        variables["full_reference"] = _adapter_full_reference_text_old(
            adapter,
            raw_example,
            reference,
        )
        return variables


class DatasetDefaultStrategy(ContextStrategy):
    """
    Exact old train-time teacher prompt.

    This is the only strategy whose behavior should bypass all new logic in
    contextualized_train_loader.py. Still, this implementation is useful for
    consistency and for manager-level calls.

    Expected behavior:
        adapter.format_train_example(raw_example)["teacher_prompt"]
    """

    name = "dataset_default"
    can_precompute = True

    def build_messages(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
        prompt_variation: int | None = None,
    ) -> list[dict[str, str]] | str:
        formatted = adapter.format_train_example(raw_example)

        if "teacher_prompt" not in formatted:
            raise KeyError(
                "adapter.format_train_example(raw_example) did not return 'teacher_prompt'. "
                f"Returned keys: {list(formatted.keys())}"
            )

        return formatted["teacher_prompt"]


class NoContextStrategy(ContextStrategy):
    """
    Teacher sees the same user task as the student.

    In training, ContextualizationManager combines this user message with the
    dataset's original prompt shell so task-specific system messages survive.
    """

    name = "no_context"
    can_precompute = True

    prompt_templates = NO_CONTEXT_PROMPT_TEMPLATES


class StaticReasoningStrategy(ContextStrategy):
    """
    Non-oracle static instruction.

    Matches analyze_contextualization.StaticReasoningStrategy. It does not use
    the reference answer.
    """

    name = "static_reasoning"
    can_precompute = True

    prompt_templates = STATIC_REASONING_PROMPT_TEMPLATES



class GoldAnswerAsExampleStrategy(ContextStrategy):
    """
    Compact gold answer shown as an example answer.

    Matches the analysis intent:
      - ToolUse: compact evaluator target / golden_answer.
      - Science: final answer inside <answer>...</answer>.
      - Other registered tasks: their adapter-defined compact answer.
    """

    name = "gold_answer_as_example"
    can_precompute = True

    prompt_templates = GOLD_ANSWER_AS_EXAMPLE_PROMPT_TEMPLATES


class GoldAnswerAsExampleV2Strategy(ContextStrategy):
    """
    Compact gold answer shown as an example answer.

    Matches the analysis intent:
      - ToolUse: compact evaluator target / golden_answer.
      - Science: final answer inside <answer>...</answer>.
      - Other registered tasks: their adapter-defined compact answer.
    """

    name = "gold_answer_as_example_v2"
    can_precompute = True

    prompt_templates = GOLD_ANSWER_AS_EXAMPLE2_PROMPT_TEMPLATES


class GoldAnswerAsGuidanceStrategy(ContextStrategy):
    """
    Compact gold answer shown as guidance rather than as an example response.

    Matches analyze_contextualization.GoldAnswerAsGuidanceStrategy.
    """

    name = "gold_answer_as_guidance"
    can_precompute = True

    prompt_templates = GOLD_ANSWER_AS_GUIDANCE_PROMPT_TEMPLATES


class GoldResponseInvStrategy(ContextStrategy):
    """
    Full gold response shown before the question, instead of after it.
    """

    name = "gold_response_inv"
    can_precompute = True
    prompt_templates = GOLD_RESPONSE_INV_PROMPT_TEMPLATES


class GoldResponseAsExampleStrategy(ContextStrategy):
    """
    Full gold response shown as an example response.

    Matches the analysis intent:
      - ToolUse: golden_response when available.
      - Science: output_text when available.
      - Other registered tasks: their adapter-defined full response.
    """

    name = "gold_response_as_example"
    can_precompute = True
    prompt_templates = GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES


class GoldResponseAsGuidanceStrategy(ContextStrategy):
    """
    Full gold response shown as guidance rather than as an example response.

    Matches analyze_contextualization.GoldResponseAsGuidanceStrategy.
    """
    name = "gold_response_as_guidance"
    can_precompute = True
    prompt_templates = GOLD_RESPONSE_AS_GUIDANCE_PROMPT_TEMPLATES


class GoldResponseAsExampleBracketStrategy(LegacyFullReferenceMixin, ContextStrategy):
    """
    Legacy/bracket ToolUse full-reference rendering.

    Use this only when you explicitly want the old ToolUse behavior where the
    full reference is rendered with the old bracket/JSON-like representation.
    """

    name = "gold_response_as_example_bracket"
    supported_datasets = {"tooluse"}
    can_precompute = True
    prompt_templates = GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES


class GoldResponseAsGuidanceBracketStrategy(LegacyFullReferenceMixin, ContextStrategy):
    """
    Legacy/bracket ToolUse full-reference rendering.

    Use this only when you explicitly want the old ToolUse behavior where the
    full reference is rendered with the old bracket/JSON-like representation.
    """

    name = "gold_response_as_guidance_bracket"
    supported_datasets = {"tooluse"}
    can_precompute = True
    prompt_templates = GOLD_RESPONSE_AS_GUIDANCE_PROMPT_TEMPLATES


class GoldHintStrategy(ContextStrategy):
    """
    Short oracle hint.

    Matches analyze_contextualization.GoldHintStrategy:
      - ToolUse uses compact natural-language action/input hints.
      - Other datasets use compact final answer hints.
    """

    name = "gold_hint"
    can_precompute = True

    def build_messages(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
        prompt_variation: int | None = None,
    ) -> list[dict[str, str]]:
        self.validate_dataset(dataset_name)
        self.resolve_prompt_variation(prompt_variation, dataset_name=dataset_name)

        question = _adapter_question_text(adapter, raw_example)

        if dataset_name == "tooluse":
            hint = _adapter_short_hint_text(adapter, raw_example, reference)
            final_instruction = """
Use this hint to choose the correct action and action input.
Do not copy a reference answer format.
Now answer the original question in the required format.
""".strip()
        else:
            hint = _adapter_compact_answer_text(adapter, raw_example, reference)
            final_instruction = """
Use this short hint to reach the correct final answer.
Do not copy any reference reasoning.
Now answer the original question in the required format.
""".strip()

        content = f"""
{question}

Short oracle hint for this instance:
{hint}

{final_instruction}
""".strip()

        return [{"role": "user", "content": content}]


class GoldHintV2Strategy(ContextStrategy):
    """
    ToolUse-only concise oracle hint.

    Matches analyze_contextualization.GoldHintV2Strategy except for removing
    accidental indentation artifacts from the triple-quoted string.
    """

    name = "gold_hint_v2"
    supported_datasets = {"tooluse"}
    can_precompute = True

    prompt_templates = GOLD_HINT_V2_PROMPT_TEMPLATES


class GoldHintV3Strategy(ContextStrategy):
    """
    ToolUse-only strict oracle hint.

    Matches analyze_contextualization.GoldHintV3Strategy except for removing
    accidental indentation artifacts from the triple-quoted string.
    """

    name = "gold_hint_v3"
    supported_datasets = {"tooluse"}
    can_precompute = True

    prompt_templates = GOLD_HINT_V3_PROMPT_TEMPLATES

# =============================================================================
# Dynamic strategies
# =============================================================================


class TooluseFailureModeStrategy(ContextStrategy):
    """
    ToolUse-specific adaptive oracle strategy.

    It looks at the student's original response, classifies the failure mode
    against golden_answer, and gives targeted feedback.

    Matches analyze_contextualization.TooluseFailureModeStrategy.
    """

    name = "tooluse_failure_mode"
    supported_datasets = {"tooluse"}
    requires_student_response = True
    requires_feedback_generation = False
    can_precompute = False

    def build_messages(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
        prompt_variation: int | None = None,
    ) -> list[dict[str, str]]:
        self.validate_dataset(dataset_name)
        self.resolve_prompt_variation(prompt_variation, dataset_name=dataset_name)

        if student_response is None:
            raise ValueError("tooluse_failure_mode requires student_response.")

        question = _adapter_question_text(adapter, raw_example)
        reference_text = _adapter_full_reference_text(adapter, raw_example, reference)
        failure = classify_tooluse_failure(adapter, student_response, reference)

        if failure.failure_mode == "correct":
            feedback = """
The previous answer already matched the required action structure.
Preserve the exact action format and avoid adding unnecessary actions or inputs.
""".strip()

        elif failure.failure_mode == "missing_action":
            feedback = f"""
The previous answer did not contain the required Action blocks.
The correct action sequence should include:
{render_reference(failure.gt_actions)}

Use explicit lines with "Action:" and "Action Input:" in the expected format.
""".strip()

        elif failure.failure_mode == "wrong_actions":
            feedback = f"""
The previous answer used the wrong tool/action sequence.

Expected action multiset:
{render_reference(failure.gt_actions)}

Previous predicted action multiset:
{render_reference(failure.pred_actions)}

Focus on choosing the exact required actions before writing the final answer.
""".strip()

        elif failure.failure_mode in {"wrong_inputs", "malformed_or_missing_inputs"}:
            feedback = f"""
The previous answer had incorrect, missing, or malformed action inputs.

Expected action input fields and values:
{render_reference(failure.gt_inputs)}

Previous predicted action inputs:
{render_reference(failure.pred_inputs)}

Focus on producing valid JSON-like Action Input blocks with the exact required fields.
""".strip()

        else:
            feedback = f"""
The previous answer did not match the expected tool-use solution.

Reference target:
{reference_text}

Focus on exact action names, exact action inputs, and the required output format.
""".strip()

        content = f"""
{question}

Targeted feedback based on the student's previous failure mode:
{feedback}

Now answer the original question again, correcting the issue while respecting the required format.
""".strip()

        return [{"role": "user", "content": content}]


class SelfFeedbackStrategy(TwoStageStrategy):
    """
    Two-stage self-feedback strategy.

    Stage 1:
        Generate feedback from:
          - original question
          - student's original answer
          - full reference/golden answer

    Stage 2:
        Final teacher prompt contains:
          - original question
          - generated feedback only

    Matches analyze_contextualization.SelfFeedbackStrategy, adapted to the
    train-time name feedback_text.
    """

    name = "self_feedback"
    requires_student_response = True
    requires_feedback_generation = True
    can_precompute = False
    default_feedback_text = "Solve carefully while respecting the required format."

    prompt_templates = SELF_FEEDBACK_PROMPT_TEMPLATES
    stage1_templates = SELF_FEEDBACK_STAGE1_TEMPLATES


class StructuredSelfFeedbackStrategy(TwoStageStrategy):
    """
    ToolUse-only structured self-feedback.

    Matches analyze_contextualization.StructuredSelfFeedbackStrategy. This is
    stricter than self_feedback and asks for exactly two concrete bullets.
    """

    name = "self_feedback_structured"
    supported_datasets = {"tooluse"}
    requires_student_response = True
    requires_feedback_generation = True
    can_precompute = False
    default_feedback_text = "Use the correct tool/action and exact required input fields."

    prompt_templates = STRUCTURED_SELF_FEEDBACK_PROMPT_TEMPLATES
    stage1_templates = STRUCTURED_SELF_FEEDBACK_STAGE1_TEMPLATES


class ConcreteSelfFeedbackStrategy(SelfFeedbackStrategy):
    """
    Same two-stage design as self_feedback, but asks the feedback generator for
    exactly two concrete, instance-specific bullet points.

    Matches analyze_contextualization.ConcreteSelfFeedbackStrategy.
    """

    name = "self_feedback_concrete"
    requires_student_response = True
    requires_feedback_generation = True
    can_precompute = False

    stage1_templates = CONCRETE_SELF_FEEDBACK_STAGE1_TEMPLATES


class RationalizationStrategy(TwoStageStrategy):
    """
    Two-step rationalization strategy.

    Stage 1:
        Generate a rationale from:
          - original question
          - compact final answer / target decision

    Stage 2:
        Final teacher prompt contains:
          - original question
          - generated rationale only

    Matches analyze_contextualization.RationalizationStrategy, adapted to the
    train-time name feedback_text.
    """

    name = "rationalization"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False
    default_feedback_text = "Think through the key reasoning needed to solve the task."

    prompt_templates = RATIONALIZATION_PROMPT_TEMPLATES

    stage1_templates = RATIONALIZATION_STAGE1_TEMPLATES


class RationalizationV2Strategy(TwoStageStrategy):
    """
    Two-step rationalization strategy.

    Stage 1:
        Generate a rationale from:
          - original question
          - compact final answer / target decision

    Stage 2:
        Final teacher prompt contains:
          - original question
          - generated rationale only

    Matches analyze_contextualization.RationalizationStrategy, adapted to the
    train-time name feedback_text.
    """

    name = "rationalization_v2"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False
    feedback_max_new_tokens = FEEDBACK_MAX_USE_DATASET_DEFAULT
    default_feedback_text = "Think through the key reasoning needed to solve the task."

    prompt_templates = RATIONALIZATION_PROMPT_TEMPLATES

    stage1_templates = RATIONALIZATION_STAGE1_TEMPLATES


class GoldResponseRewriteAsExampleStrategy(TwoStageStrategy):
    """
    Two-stage cleaned-reference example strategy.

    Stage 1:
        Rewrite the full gold/reference response in the model's own words,
        while preserving the final answer and matching the original task's
        required format.

    Stage 2:
        Show the rewritten reference response as a clean example, then ask the
        teacher to answer the original task.

    Motivation
    ----------
    This strategy is designed for datasets such as Science, where raw gold
    responses can be useful but may also induce excessive length, brittle
    rationale imitation, or stylistic artifacts.  The stage-1 rewrite acts as a
    normalization step before the answer is used as an example.

    It differs from:
      - gold_response_as_example_fix: raw gold response is shown directly.
      - gold_response_as_guidance_fix: reference is shown as guidance, not as an
        example response.
      - rationalization: stage 1 produces only a rationale, not a full
        answer-like reference.
    """

    name = "gold_response_rewrite_as_example"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False
    default_feedback_text = (
        "Provide a clean, concise, task-aligned answer that preserves the "
        "reference answer and required output format."
    )

    prompt_templates = GOLD_RESPONSE_REWRITE_AS_EXAMPLE_PROMPT_TEMPLATES
    stage1_templates = GOLD_RESPONSE_REWRITE_STAGE1_TEMPLATES


class GoldResponseRewriteAsExampleV2Strategy(TwoStageStrategy):
    """
    Two-stage cleaned-reference example strategy.

    Stage 1:
        Rewrite the full gold/reference response in the model's own words,
        while preserving the final answer and matching the original task's
        required format.

    Stage 2:
        Show the rewritten reference response as a clean example, then ask the
        teacher to answer the original task.

    Motivation
    ----------
    This strategy is designed for datasets such as Science, where raw gold
    responses can be useful but may also induce excessive length, brittle
    rationale imitation, or stylistic artifacts.  The stage-1 rewrite acts as a
    normalization step before the answer is used as an example.

    It differs from:
      - gold_response_as_example_fix: raw gold response is shown directly.
      - gold_response_as_guidance_fix: reference is shown as guidance, not as an
        example response.
      - rationalization: stage 1 produces only a rationale, not a full
        answer-like reference.
    """

    name = "gold_response_rewrite_as_example_v2"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False
    default_feedback_text = (
        "Provide a clean, concise, task-aligned answer that preserves the "
        "reference answer and required output format."
    )

    prompt_templates = GOLD_RESPONSE_REWRITE_AS_EXAMPLE_V2_PROMPT_TEMPLATES
    stage1_templates = GOLD_RESPONSE_REWRITE_V2_STAGE1_TEMPLATES


class GoldResponseRewriteAsExampleV3Strategy(TwoStageStrategy):
    """
    Two-stage cleaned-reference example strategy.

    Stage 1:
        Rewrite the full gold/reference response in the model's own words,
        while preserving the final answer and matching the original task's
        required format.

    Stage 2:
        Show the rewritten reference response as a clean example, then ask the
        teacher to answer the original task.

    Motivation
    ----------
    This strategy is designed for datasets such as Science, where raw gold
    responses can be useful but may also induce excessive length, brittle
    rationale imitation, or stylistic artifacts.  The stage-1 rewrite acts as a
    normalization step before the answer is used as an example.

    It differs from:
      - gold_response_as_example_fix: raw gold response is shown directly.
      - gold_response_as_guidance_fix: reference is shown as guidance, not as an
        example response.
      - rationalization: stage 1 produces only a rationale, not a full
        answer-like reference.
    """

    name = "gold_response_rewrite_as_example_v3"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False
    feedback_max_new_tokens = FEEDBACK_MAX_USE_DATASET_DEFAULT
    default_feedback_text = (
        "Provide a clean, concise, task-aligned answer that preserves the "
        "reference answer and required output format."
    )

    prompt_templates = GOLD_RESPONSE_REWRITE_AS_EXAMPLE_V2_PROMPT_TEMPLATES
    stage1_templates = GOLD_RESPONSE_REWRITE_V2_STAGE1_TEMPLATES


class SiblingResponseAsExampleStrategy(ContextStrategy):
    """
    Correct sibling response shown through the existing gold-response-as-example
    prompt templates.

    This strategy does not use privileged information to generate the sibling.
    At runtime, DistilTrainer samples extra completions from the original
    student prompt, scores them with the dataset adapter, and passes the first
    correct usable sibling as sibling_text.

    We then substitute that selected sibling into the existing
    GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES by overriding $full_reference.
    If no correct sibling is found, sibling_text is the adapter-provided
    fallback full gold response.
    """

    name = "sibling_response_as_example"
    requires_student_response = False
    requires_feedback_generation = False
    requires_sibling_selection = True
    can_precompute = False

    prompt_templates = GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES

    def build_template_variables(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
    ) -> dict[str, Any]:
        variables = super().build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
        )

        selected_response = str(sibling_text or "").strip()
        if not selected_response:
            selected_response = adapter.get_fallback_response_for_context(
                raw_example,
                reference,
            )

        # Reuse GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES unchanged by making
        # the selected sibling/fallback behave as the template's $full_reference.
        variables["full_reference"] = selected_response
        variables["sibling_text"] = selected_response

        return variables
    

class SdpoSiblingFeedbackStrategy(TwoStageStrategy):
    """
    SDPO-like self-teacher strategy.

    Runtime behavior:
      - The ordinary student rollout is the response whose log-probabilities
        are re-evaluated.
      - If that rollout is correct, it is used as the correct solution in the
        self-teacher prompt, and feedback is skipped.
      - Otherwise, the trainer samples unprivileged sibling rollouts from the
        same original prompt and selects the first correct usable sibling.
      - If a correct sibling exists, it is used as the correct solution, and
        feedback is skipped.
      - If no correct solution is available, privileged concrete feedback is
        generated for the unsuccessful original rollout and inserted instead.

    This follows the SDPO-style conditional structure: use a correct solution
    when available; otherwise condition the self-teacher on feedback.
    """

    name = "sdpo_sibling_feedback"
    requires_student_response = True
    requires_feedback_generation = True
    requires_sibling_selection = True
    can_precompute = False

    # SDPO-like behavior: do not inject gold as a fake successful rollout.
    sibling_fallback_to_gold = False

    # SDPO-like routing: avoid unnecessary sibling/feedback generation.
    sibling_only_if_student_incorrect = True
    feedback_only_without_correct_solution = True

    default_feedback_text = ""

    stage1_templates = CONCRETE_SELF_FEEDBACK_STAGE1_TEMPLATES
    prompt_templates = SDPO_SIBLING_FEEDBACK_PROMPT_TEMPLATES

    def build_template_variables(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
    ) -> dict[str, Any]:
        variables = super().build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
        )

        student_response_text = str(student_response or "").strip()
        sibling_response_text = str(sibling_text or "").strip()
        feedback_text = str(feedback_text or "").strip()

        student_is_correct = False
        if student_response_text:
            scores = adapter.score_responses_for_context(
                [student_response_text],
                raw_example=raw_example,
                reference=reference,
            )
            student_is_correct = bool(scores and int(scores[0]) == 1)

        if student_is_correct:
            solution_text = student_response_text
            feedback_text = ""
        elif sibling_response_text:
            solution_text = sibling_response_text
            feedback_text = ""
        else:
            solution_text = ""

        if solution_text:
            variables["sdpo_solution_block"] = (
                "Correct solution:\n"
                f"{solution_text}\n\n"
            )
        else:
            variables["sdpo_solution_block"] = ""

        if feedback_text:
            variables["sdpo_feedback_block"] = (
                "The following is feedback from your unsuccessful earlier attempt:\n"
                f"{feedback_text}\n\n"
            )
        else:
            variables["sdpo_feedback_block"] = ""

        return variables

# =============================================================================
# Registry
# =============================================================================


STRATEGY_REGISTRY: dict[str, ContextStrategy] = {
    # Dataset-defined default teacher prompt.
    "dataset_default": DatasetDefaultStrategy(),

    # Baselines / controls.
    "no_context": NoContextStrategy(),
    "static_reasoning": StaticReasoningStrategy(),

    # Oracle ablations.
    "gold_answer_as_example": GoldAnswerAsExampleStrategy(),
    "gold_answer_as_example_v2": GoldAnswerAsExampleV2Strategy(),
    "gold_answer_as_guidance": GoldAnswerAsGuidanceStrategy(),
    "gold_response_inv": GoldResponseInvStrategy(),
    "gold_response_as_example": GoldResponseAsExampleStrategy(),
    "gold_response_as_guidance": GoldResponseAsGuidanceStrategy(),
    "gold_response_as_example_bracket": GoldResponseAsExampleBracketStrategy(),
    "gold_response_as_guidance_bracket": GoldResponseAsGuidanceBracketStrategy(),
    "gold_response_rewrite_as_example": GoldResponseRewriteAsExampleStrategy(),
    "gold_response_rewrite_as_example_v2": GoldResponseRewriteAsExampleV2Strategy(),
    "gold_response_rewrite_as_example_v3": GoldResponseRewriteAsExampleV3Strategy(),
    "gold_hint": GoldHintStrategy(),
    "gold_hint_v2": GoldHintV2Strategy(),
    "gold_hint_v3": GoldHintV3Strategy(),

    # Adaptive / dynamic strategies.
    "tooluse_failure_mode": TooluseFailureModeStrategy(),
    "self_feedback": SelfFeedbackStrategy(),
    "self_feedback_structured": StructuredSelfFeedbackStrategy(),
    "self_feedback_concrete": ConcreteSelfFeedbackStrategy(),

    # Two-step rationale strategy.
    "rationalization": RationalizationStrategy(),
    "rationalization_v2": RationalizationV2Strategy(),

    # RL-like strategies
    "sibling_response_as_example": SiblingResponseAsExampleStrategy(),
    "sdpo_sibling_feedback": SdpoSiblingFeedbackStrategy(),
}


def get_context_strategy(name: str) -> ContextStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown context strategy {name!r}. "
            f"Available strategies: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


def resolve_strategies(strategy_arg: str, dataset_name: str) -> list[ContextStrategy]:
    """
    Optional convenience helper matching analyze_contextualization.resolve_strategies.

    This is not required by main.py if you use argparse choices, but it is useful
    for tests and for any future script that accepts comma-separated strategies
    or "all".
    """
    if strategy_arg == "all":
        return [
            strategy
            for strategy in STRATEGY_REGISTRY.values()
            if strategy.supported_datasets is None
            or dataset_name in strategy.supported_datasets
        ]

    names = [x.strip() for x in strategy_arg.split(",") if x.strip()]
    unknown = [name for name in names if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown strategy name(s): {unknown}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}, or 'all'."
        )

    strategies = [STRATEGY_REGISTRY[name] for name in names]

    unsupported = [
        strategy.name
        for strategy in strategies
        if strategy.supported_datasets is not None
        and dataset_name not in strategy.supported_datasets
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported strategies for dataset={dataset_name}: {unsupported}"
        )

    return strategies
