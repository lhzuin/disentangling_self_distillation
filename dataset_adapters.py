# dataset_adapters.py
"""Dataset registry, training adapters, and evaluation/scoring contracts.

Each ``DatasetAdapter`` owns task-specific data paths, train/eval formatting,
response scoring, generation limits, response-record serialization, and hooks
used by contextualization strategies. ``DatasetSpec`` stores dataset-level
analysis metadata such as expert/reference accuracy and chance floor.

Base-model accuracies are model-specific and are resolved from
``model_registry.py`` by ``get_dataset_metric_defaults``. The adapter layer does
not duplicate those values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template
from typing import Any
import json
import re
from collections import Counter

import numpy as np
from datasets import load_from_disk
from data_utils.math_verify_scorer import OpenR1MathVerifyScorer
from data_utils.base_notation_math_scorer import (
    BaseAwareMathVerifier,
    BaseAwareVerificationResult,
)
from model_registry import DEFAULT_MODEL_KEY, resolve_model_spec

from data_utils.spatial_contradiction2.spatial_contradiction_core import (
    SpatialCoordinateVerifier,
    CoordinateVerificationResult,
)

@dataclass
class EvalOutput:
    prompts: list[str]
    references: list[Any]
    raw_examples: list[dict]


class DatasetAdapter(ABC):
    name: str = ""
    train_path: str = ""
    eval_path: str = ""
    default_max_new_tokens: int = 1024
    default_max_model_len: int | None = None
    results_filename: str = "eval_results.json"
    responses_filename: str = "eval_responses.json"

    @abstractmethod
    def format_train_example(self, example: dict) -> dict:
        pass

    @abstractmethod
    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        pass

    @abstractmethod
    def score_responses(self, responses: list[str], references: list[Any]) -> list[int]:
        pass

    @abstractmethod
    def build_response_records(
        self,
        prompts: list[str],
        responses: list[str],
        references: list[Any],
        scores: list[int],
        raw_examples: list[dict],
    ) -> list[dict]:
        pass

    def get_train_path(self, train_view: str | None = None) -> str:
        """Resolve a named canonical training view.

        Existing adapters expose only ``train_path`` and therefore support the
        implicit ``default`` view exactly as before. Datasets with multiple
        canonical training representations may define ``train_views``.
        """
        view = "default" if train_view in (None, "", "default") else str(train_view)
        if view == "default":
            if not self.train_path:
                raise ValueError(f"Adapter {self.name!r} does not define train_path")
            return self.train_path

        train_views = getattr(self, "train_views", None) or {}
        if view not in train_views:
            available = ["default", *sorted(train_views)]
            raise ValueError(
                f"Dataset {self.name!r} does not provide train view {view!r}. "
                f"Available views: {available}"
            )
        return str(train_views[view])

    def load_train(self, seed: int = 42, train_view: str | None = None):
        dataset = load_from_disk(self.get_train_path(train_view))
        dataset = dataset.map(self.format_train_example, remove_columns=dataset.column_names)
        dataset = dataset.shuffle(seed=seed)
        return dataset, None

    def load_eval(self):
        return load_from_disk(self.eval_path)

    def evaluate(self, responses: list[str], references: list[Any]) -> dict:
        scores = self.score_responses(responses, references)
        accuracy = float(np.mean(scores)) if scores else 0.0
        return {
            "accuracy": accuracy,
            "num_correct": int(sum(scores)),
            "num_total": len(scores),
            "per_sample_scores": scores,
        }

    def build_train_analysis_output(self, train_data, tokenizer) -> EvalOutput:
        """
        Build prompts/references/raw_examples for analysing the train split.

        Default behavior:
        use format_train_example(example)["prompt"] for student prompts,
        and use format_train_example(example)["target"] as reference.

        Datasets can override this if their evaluator expects something different.
        """
        raw_examples = train_data.to_list()

        prompts = []
        references = []

        for ex in raw_examples:
            formatted = self.format_train_example(ex)

            prompt_obj = formatted["prompt"]
            if isinstance(prompt_obj, str):
                prompt_text = prompt_obj
            else:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_obj,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompts.append(prompt_text)
            references.append(formatted["target"])

        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )
    
    def load_raw_train(self, train_view: str | None = None):
        """Load a canonical raw training view before adapter formatting."""
        return load_from_disk(self.get_train_path(train_view))
    
    def get_prompt_messages(self, raw_example: dict) -> list[dict[str, str]] | None:
        """
        Return the original chat-message prompt shell when available.

        This preserves dataset-specific system instructions, formatting rules,
        tool documentation, etc.

        Subclasses can override this if their schema is unusual.
        """
        for key in ["prompt", "messages"]:
            value = raw_example.get(key)
            if (
                isinstance(value, list)
                and all(
                    isinstance(x, dict)
                    and "role" in x
                    and "content" in x
                    for x in value
                )
            ):
                return [dict(x) for x in value]

        return None

    def contextualize_messages(
        self,
        raw_example: dict,
        strategy_messages: list[dict[str, str]] | str,
    ) -> list[dict[str, str]] | str:
        """
        Combine a strategy-generated contextualized user prompt with the
        dataset's original prompt shell.

        This prevents strategies from dropping dataset system messages that
        enforce task-specific answer formats.

        Behavior:
          - If the raw example has chat messages, preserve all non-final-user
            messages and replace the final user message with the strategy's
            final user content.
          - If no chat shell exists, return the strategy messages unchanged.
        """
        base_messages = self.get_prompt_messages(raw_example)

        if base_messages is None:
            return strategy_messages

        if isinstance(strategy_messages, str):
            strategy_user_content = strategy_messages
        else:
            strategy_user_messages = [
                msg
                for msg in strategy_messages
                if isinstance(msg, dict) and msg.get("role") == "user"
            ]

            if strategy_user_messages:
                strategy_user_content = str(strategy_user_messages[-1].get("content", ""))
            else:
                strategy_user_content = "\n\n".join(
                    str(msg.get("content", ""))
                    for msg in strategy_messages
                    if isinstance(msg, dict)
                ).strip()

        contextualized = [dict(msg) for msg in base_messages]

        for i in range(len(contextualized) - 1, -1, -1):
            if contextualized[i].get("role") == "user":
                contextualized[i]["content"] = strategy_user_content
                return contextualized

        contextualized.append({"role": "user", "content": strategy_user_content})
        return contextualized

    def _render_reference_for_context(self, value: Any) -> str:
        """
        Render arbitrary gold/reference objects as readable text for prompts.

        This is intentionally adapter-local, so analysis strategies do not need
        to know whether a dataset stores answers as strings, dicts, or lists.
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                return "\n".join(value)

        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            return str(value)
    
    def _render_reference_for_context_old(self, value: Any) -> str:
        """
        Compatibility renderer for strategies that intentionally use JSON-style
        rendering of list-valued full responses. In ToolUse this preserves the
        bracketed ``golden_response`` representation required by those strategy
        definitions.
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            return str(value)

    def extract_xml_answer_text(self, text: Any) -> str:
        """
        Extract the text inside the final <answer>...</answer> block.

        Used by answer-only contextualization strategies for datasets whose
        training targets are full XML-style responses.
        """
        if text is None:
            return ""

        text = str(text)
        matches = re.findall(
            r"<answer>\s*(.*?)\s*</answer>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if matches:
            return matches[-1].strip()

        # Conservative fallback for malformed but common partial outputs.
        m = re.search(r"<answer>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        return ""

    def get_gold_answer_text(self, raw_example: dict, reference: Any) -> str:
        """
        Return the compact final answer / evaluator target for this example.

        This is the method that answer-only strategies should use.

        Default order:
          1. final answer inside raw_example["output_text"], if present;
          2. explicit answer-like raw fields;
          3. the adapter-provided reference.

        Subclasses should override this when their schema has a cleaner target.
        """
        if "output_text" in raw_example:
            answer = self.extract_xml_answer_text(raw_example.get("output_text"))
            if answer:
                return answer

        for key in ["answer_letter", "answer_text", "answer", "target", "golden_answer"]:
            value = raw_example.get(key)
            if value not in [None, ""]:
                return self._render_reference_for_context(value)

        return self._render_reference_for_context(reference)

    def get_gold_response_text(self, raw_example: dict, reference: Any) -> str:
        """
        Return the full gold response / full reference solution when available.

        This is the method that full-response strategies should use.

        Default order:
          1. output_text;
          2. golden_response;
          3. reasoning + answer when both exist;
          4. compact answer fallback.
        """
        if raw_example.get("output_text"):
            return self._render_reference_for_context(raw_example["output_text"])

        if raw_example.get("golden_response"):
            return self._render_reference_for_context(raw_example["golden_response"])

        reasoning = raw_example.get("reasoning")
        answer = raw_example.get("answer")
        if reasoning and answer:
            return f"{self._render_reference_for_context(reasoning)}\n\nAnswer:\n{self._render_reference_for_context(answer)}"

        return self.get_gold_answer_text(raw_example, reference)

    def get_question_text(self, raw_example: dict) -> str:
        """
        Return the user-facing question/task text.

        Strategies should call this instead of inspecting dataset-specific
        fields directly. Subclasses can override this when needed.
        """
        messages = self.get_prompt_messages(raw_example)
        if messages is not None:
            user_messages = [
                str(msg.get("content", ""))
                for msg in messages
                if isinstance(msg, dict) and msg.get("role") == "user"
            ]
            if user_messages:
                return user_messages[-1]

        for key in ["prompt", "question", "input", "content"]:
            value = raw_example.get(key)
            if value not in [None, ""]:
                return self._render_reference_for_context(value)

        return self._render_reference_for_context(raw_example)

    def get_reference_for_context(self, raw_example: dict, formatted_example: dict | None = None) -> Any:
        """
        Return the reference object that strategies should use.

        This is intentionally different from target:
          - target is the sequence used as dataset completion target;
          - reference_for_context is the compact or full object used to build
            contextualized teacher prompts.

        Default fallback is formatted_example['target'].
        """
        if formatted_example is not None and "target" in formatted_example:
            return formatted_example["target"]
        return raw_example.get("target", raw_example)

    def get_full_reference_text_for_context(self, raw_example: dict, reference: Any) -> str:
        """
        Full reference solution/response used by full-response and feedback strategies.
        """
        return self.get_gold_response_text(raw_example, reference)
    
    def get_full_reference_text_for_context_old(self, raw_example: dict, reference: Any) -> str:
        """
        Legacy full-reference rendering.

        Mirrors get_full_reference_text_for_context, but uses the legacy renderer
        when the reference object must be rendered directly.
        """
        for key in ("output_text", "golden_response", "reference", "answer"):
            value = raw_example.get(key)
            if value:
                return self._render_reference_for_context_old(value)

        return self._render_reference_for_context_old(reference)

    def get_compact_reference_text_for_context(self, raw_example: dict, reference: Any) -> str:
        """
        Compact final answer used by answer-only/rationale strategies.
        """
        return self.get_gold_answer_text(raw_example, reference)

    def get_short_hint_text_for_context(self, raw_example: dict, reference: Any) -> str:
        """
        Short oracle hint used by GoldHintStrategy.

        Default behavior is the compact final answer. Datasets can override
        this when a more useful compact hint exists.
        """
        return self.get_compact_reference_text_for_context(raw_example, reference)

    def get_source_context_text_for_context(self, raw_example: dict) -> str:
        """Return optional source text used by source-grounded strategies.

        Most datasets do not expose a separate source document and therefore
        return an empty string. Multi-view datasets can override this hook.
        """
        return ""

    def get_scoring_reference_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> Any:
        """
        Return the reference object expected by score_responses when scoring
        generated context candidates.

        This may differ from the full response used for prompting. For example,
        Science uses a full gold response for contextualization but its scorer
        expects only the compact option letter.
        """
        return reference

    def score_responses_for_context(
        self,
        responses: list[str],
        raw_example: dict,
        reference: Any,
    ) -> list[int]:
        """
        Score generated context candidates for one raw example.

        The default implementation reuses the dataset evaluator by repeating
        the adapter-specific scoring reference once per candidate.
        """
        if not responses:
            return []

        scoring_reference = self.get_scoring_reference_for_context(
            raw_example,
            reference,
        )
        return self.score_responses(
            responses,
            [scoring_reference for _ in responses],
        )
    
    def _balanced_xml_tags(self, text: str) -> set[str]:
        """
        Return XML-like tags that have both opening and closing forms.

        This is intentionally lightweight. It is used only to avoid selecting
        structurally incomplete examples, such as answer-only outputs, when the
        full reference response contains required tagged sections.
        """
        text = str(text or "")

        opened = set(
            re.findall(
                r"<([A-Za-z][A-Za-z0-9_-]*)\b[^>/]*>",
                text,
                flags=re.IGNORECASE,
            )
        )
        closed = set(
            re.findall(
                r"</([A-Za-z][A-Za-z0-9_-]*)\s*>",
                text,
                flags=re.IGNORECASE,
            )
        )

        return {tag.lower() for tag in opened & closed}

    def is_usable_context_response_example(
        self,
        response: str,
        raw_example: dict,
        reference: Any,
    ) -> bool:
        """
        Return whether a correct generated response is structurally safe to use
        as an in-context example.

        Correctness alone is not always enough: a candidate may have the right
        final answer but be answer-only or structurally incomplete. When the
        full reference response contains balanced XML-like tags, require the
        generated candidate to preserve those tags.
        """
        response = str(response or "").strip()
        if not response:
            return False

        full_reference = self.get_full_reference_text_for_context(
            raw_example,
            reference,
        )

        required_tags = self._balanced_xml_tags(full_reference)
        if not required_tags:
            return True

        response_tags = self._balanced_xml_tags(response)
        return required_tags.issubset(response_tags)

    def get_fallback_response_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> str:
        """
        Fallback response used when no correct generated sibling is found.
        """
        return self.get_full_reference_text_for_context(raw_example, reference)

class ToolUseAdapter(DatasetAdapter):
    name = "tooluse"
    train_path = "data/tooluse_data/train_data"
    eval_path = "data/tooluse_data/eval_data"
    default_max_new_tokens = 1024
    default_max_model_len = None
    results_filename = "eval_tooluse_results.json"
    responses_filename = "eval_tooluse_responses.json"

    def format_train_example(self, example: dict) -> dict:
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [{
                "role": "user",
                "content": teacher_prompt.substitute(
                    orig_content=example["prompt"],
                    output_text="\n".join(example["golden_response"]),
                ),
            }],
            "target": "\n".join(example["golden_response"]),
        }

    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        raw_examples = eval_data.to_list()

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in raw_examples
        ]
        references = [ex["golden_answer"] for ex in raw_examples]

        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )

    def extract_actions(self, text: str) -> list[str]:
        return re.findall(r"Action:\s*(\w+)", text)
    
    def extract_actions_new(self, text: str) -> list[str]:
        return [
            action.strip()
            for action in re.findall(r"Action:\s*([^\r\n]+)", text)
        ]

    def extract_action_inputs(self, text: str) -> dict:
        json_blocks = re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL)
        combined_dict = {}
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                combined_dict.update(parsed)
            except json.JSONDecodeError:
                continue
        return combined_dict
    
    def extract_action_inputs_new(self, text: str) -> dict: #TODO: start using this new method and batch rescore tooluse
        """
        Extract Action Input JSON objects from a model response.

        Preserves the old evaluator behavior:
        - find every `Action Input:` block;
        - parse each JSON object;
        - merge parsed dictionaries into one combined dict;
        - ignore malformed blocks.

        Improvement over the old regex:
        - supports nested JSON objects, e.g.
            {"headers": {"Content-Type": "application/json"}, "data": {...}}
        """
        json_blocks = self.extract_action_input_json_blocks(text)

        combined_dict = {}
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    combined_dict.update(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return combined_dict


    def extract_action_input_json_blocks(self, text: str) -> list[str]:
        """
        Extract complete JSON objects after each `Action Input:` marker.

        Uses json.JSONDecoder.raw_decode instead of a non-greedy regex, so nested
        JSON objects are handled correctly.
        """
        blocks = []
        marker = "Action Input:"
        decoder = json.JSONDecoder()
        start = 0

        while True:
            marker_pos = text.find(marker, start)
            if marker_pos == -1:
                break

            pos = marker_pos + len(marker)

            while pos < len(text) and text[pos].isspace():
                pos += 1

            # Keep the old contract: Action Input should be a JSON object.
            if pos >= len(text) or text[pos] != "{":
                start = pos + 1
                continue

            try:
                _parsed, end_rel = decoder.raw_decode(text[pos:])
                blocks.append(text[pos : pos + end_rel])
                start = pos + end_rel
            except json.JSONDecodeError:
                start = pos + 1

        return blocks

    def score_responses(self, responses: list[str], references: list[list[dict]]) -> list[int]:
        results = []

        for response, golden_answer in zip(responses, references):
            #pred_actions = self.extract_actions(response)
            #pred_inputs = self.extract_action_inputs(response)
            pred_actions = self.extract_actions_new(response)
            pred_inputs = self.extract_action_inputs_new(response)

            gt_actions = [item["Action"] for item in golden_answer]
            gt_inputs = {}
            for item in golden_answer:
                try:
                    gt_inputs.update(json.loads(item["Action_Input"]))
                except Exception:
                    pass

            actions_match = Counter(pred_actions) == Counter(gt_actions)
            inputs_match = pred_inputs == gt_inputs
            results.append(1 if (actions_match and inputs_match) else 0)

        return results

    def build_response_records(
        self,
        prompts: list[str],
        responses: list[str],
        references: list[list[dict]],
        scores: list[int],
        raw_examples: list[dict],
    ) -> list[dict]:
        return [
            {
                "prompt": prompts[i],
                "response": responses[i],
                "golden_answer": references[i],
                "correct": bool(scores[i]),
            }
            for i in range(len(responses))
        ]
    
    def build_train_analysis_output(self, train_data, tokenizer) -> EvalOutput:
        raw_examples = train_data.to_list()

        prompts = [
            tokenizer.apply_chat_template(
                self.format_train_example(ex)["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in raw_examples
        ]

        references = [ex["golden_answer"] for ex in raw_examples]

        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )
    
    def get_prompt_messages(self, raw_example: dict) -> list[dict[str, str]] | None:
        prompt = raw_example.get("prompt")
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return super().get_prompt_messages(raw_example)

    def get_gold_answer_text(self, raw_example: dict, reference: Any) -> str:
        """
        Compact ToolUse gold answer.

        For ToolUse, the compact evaluator target is the structured
        golden_answer: a list of required Action / Action_Input objects.

        This intentionally does NOT use golden_response, because golden_response
        is a full trace with reasoning text.
        """
        if "golden_answer" in raw_example:
            return self._render_reference_for_context(raw_example["golden_answer"])
        return self._render_reference_for_context(reference)

    def get_gold_response_text(self, raw_example: dict, reference: Any) -> str:
        """
        Full ToolUse gold response.

        For ToolUse, this is golden_response when available. On eval split,
        golden_response may be absent, so we fall back to the compact answer.
        """
        if raw_example.get("golden_response"):
            return self._render_reference_for_context(raw_example["golden_response"])
        return self.get_gold_answer_text(raw_example, reference)
    
    def get_gold_response_text_old(self, raw_example: dict, reference: Any) -> str:
        """
        Compatibility full ToolUse gold response.

        Renders ``golden_response`` list values as JSON, including brackets, for
        strategies that explicitly use this representation.
        """
        if raw_example.get("golden_response"):
            return self._render_reference_for_context_old(raw_example["golden_response"])

        return self.get_gold_answer_text(raw_example, reference)


    def get_full_reference_text_for_context_old(self, raw_example: dict, reference: Any) -> str:
        """
        Legacy full-reference text for ToolUse.
        """
        return self.get_gold_response_text_old(raw_example, reference)

    def get_reference_for_context(self, raw_example: dict, formatted_example: dict | None = None) -> Any:
        """
        For ToolUse, strategies should usually see the structured evaluator
        target, not only the full golden_response trace.
        """
        if "golden_answer" in raw_example:
            return raw_example["golden_answer"]
        return super().get_reference_for_context(raw_example, formatted_example)

    def get_question_text(self, raw_example: dict) -> str:
        return str(raw_example.get("prompt", ""))
    
    def parse_tooluse_reference_for_hint(self, reference: Any) -> list[dict[str, Any]]:
        normalized = []

        if not isinstance(reference, list):
            return normalized

        for item in reference:
            if not isinstance(item, dict):
                continue

            action = item.get("Action")
            action_input = item.get("Action_Input", {})

            if isinstance(action_input, str):
                try:
                    parsed_input = json.loads(action_input)
                except Exception:
                    parsed_input = action_input
            else:
                parsed_input = action_input

            normalized.append(
                {
                    "Action": action,
                    "Action_Input": parsed_input,
                }
            )

        return normalized
    
    def render_tooluse_action_hint_for_context(self, reference: Any) -> str:
        normalized = self.parse_tooluse_reference_for_hint(reference)

        if not normalized:
            try:
                return json.dumps(reference, indent=2, ensure_ascii=False)
            except Exception:
                return str(reference)

        parts = []
        for step_id, step in enumerate(normalized, start=1):
            action = step.get("Action")
            action_input = step.get("Action_Input")

            parts.append(f"Step {step_id}:")
            parts.append(f"- Correct tool/action: {action}")
            parts.append("- Required action input:")

            # if isinstance(action_input, dict):
            #     if action_input:
            #         for key, value in action_input.items():
            #             parts.append(f"  - {key}: {self._render_reference_for_context(value)}")
            #             #parts.append(f"  - {key}: {value}")
            #     else:
            #         parts.append("  - {}")
            # else:
            #     #parts.append(f"  - {action_input}")
            #     parts.append(f"  - {self._render_reference_for_context(action_input)}")
            if isinstance(action_input, dict):
                if action_input:
                    for key, value in action_input.items():
                        parts.append(f"  - {key}: {self._render_reference_for_context(value)}")
                else:
                    parts.append("  - {}")
            else:
                parts.append(f"  - {self._render_reference_for_context(action_input)}")

        return "\n".join(parts)

    def get_short_hint_text_for_context(self, raw_example: dict, reference: Any) -> str:
        if "golden_answer" in raw_example:
            return self.render_tooluse_action_hint_for_context(raw_example["golden_answer"])
        return self.render_tooluse_action_hint_for_context(reference)

class ScienceAdapter(DatasetAdapter):
    name = "science"
    train_path = "data/science_data/train_data"
    eval_path = "data/science_data/eval_data"
    default_max_new_tokens = 2048
    default_max_model_len = 4096
    results_filename = "eval_science_results.json"
    responses_filename = "eval_science_responses.json"

    def format_train_example(self, example: dict) -> dict:
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {
                    "role": "user",
                    "content": teacher_prompt.substitute(
                        orig_content=example["messages"][1]["content"],
                        output_text=example["output_text"],
                    ),
                },
            ],
            "target": example["output_text"],
        }

    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        raw_examples = eval_data.to_list()

        prompts = [
            tokenizer.apply_chat_template(
                ex["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in raw_examples
        ]
        references = [ex["answer"] for ex in raw_examples]

        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )

    def extract_xml_answer(self, text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def score_responses(self, responses: list[str], references: list[str]) -> list[int]:
        results = []
        for response, answer in zip(responses, references):
            extracted = self.extract_xml_answer(response)
            results.append(1 if extracted == answer else 0)
        return results

    def build_response_records(
        self,
        prompts: list[str],
        responses: list[str],
        references: list[str],
        scores: list[int],
        raw_examples: list[dict],
    ) -> list[dict]:
        return [
            {
                "prompt": raw_examples[i]["prompt"],
                "response": responses[i],
                "answer": references[i],
                "correct": bool(scores[i]),
            }
            for i in range(len(responses))
        ]
    
    def build_train_analysis_output(self, train_data, tokenizer) -> EvalOutput:
        raw_train_examples = train_data.to_list()

        prompts = [
            tokenizer.apply_chat_template(
                self.format_train_example(ex)["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in raw_train_examples
        ]

        references = [
            self.extract_xml_answer(ex["output_text"])
            for ex in raw_train_examples
        ]

        # Make train examples compatible with the existing eval-style
        # build_response_records contract, without changing build_response_records.
        raw_examples = [
            {
                **ex,
                "prompt": ex["messages"],
                "answer": self.extract_xml_answer(ex["output_text"]),
            }
            for ex in raw_train_examples
        ]

        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )

    def get_gold_answer_text(self, raw_example: dict, reference: Any) -> str:
        """
        Compact Science gold answer.

        For Science, this is the content inside <answer>...</answer>, usually
        a single option letter A/B/C/D.
        """
        if "output_text" in raw_example:
            answer = self.extract_xml_answer(raw_example["output_text"])
            if answer:
                return answer

        # Eval/build_train_analysis references are already compact letters.
        return self._render_reference_for_context(reference)

    def get_gold_response_text(self, raw_example: dict, reference: Any) -> str:
        """
        Full Science gold response.

        For train split, this is output_text, including reasoning and answer.
        For eval split, output_text may be absent, so we fall back to the compact
        answer.
        """
        if raw_example.get("output_text"):
            return self._render_reference_for_context(raw_example["output_text"])

        return self.get_gold_answer_text(raw_example, reference)

    def get_scoring_reference_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> str:
        """
        Science score_responses expects the compact option letter, while the
        contextualization reference is usually the full output_text.
        """
        return self.get_gold_answer_text(raw_example, reference)


class MathContradictionAdapter(DatasetAdapter):
    """Adapter for the audited alternate-base math contradiction dataset.

    The stored prompt and gold solution remain entirely in the alternate-base
    world. Evaluation back-converts only the compact gold answer and the final
    ``\\boxed{...}`` prediction to decimal before delegating to the shared
    OpenR1 Math-Verify scorer.

    This adapter has no hidden-thinking or target-mode switches:
    ``output_text`` is always the public training target.
    The prompt intentionally does not reveal the base, preserving the intended
    knowledge-acquisition task.
    """

    name = "math_contradiction"
    train_path = "data/math_contradiction_data/train_data"
    eval_path = "data/math_contradiction_data/eval_data"

    default_base = 9
    default_max_new_tokens = 1024
    default_max_model_len = 4096
    results_filename = "eval_math_contradiction_results.json"
    responses_filename = "eval_math_contradiction_responses.json"

    MATH_USER_PROMPT_TEMPLATE = (
        "You will be given a problem.\n"
        "Please reason step by step, and put your final answer within "
        "\\boxed{{}}:\n"
        "{problem}"
    )
    DATASET_DEFAULT_TEACHER_TEMPLATE = """
{question}

Here is a verified reference solution for the same problem:
{reference_response}

Now solve the original problem yourself. Keep the requested final-answer format.
""".strip()

    _verifier: OpenR1MathVerifyScorer | None = None
    _base_verifiers: dict[int, BaseAwareMathVerifier] = {}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _ensure_messages(cls, example: dict[str, Any]) -> list[dict[str, str]]:
        messages = example.get("messages")
        if (
            isinstance(messages, list)
            and messages
            and all(
                isinstance(message, dict)
                and "role" in message
                and "content" in message
                for message in messages
            )
        ):
            return [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in messages
            ]

        problem = cls._normalize_text(
            example.get("problem") or example.get("question") or ""
        )
        return [
            {
                "role": "user",
                "content": cls.MATH_USER_PROMPT_TEMPLATE.format(problem=problem),
            }
        ]

    @classmethod
    def _get_answer(cls, example: dict[str, Any]) -> str:
        for key in ("answer", "golden_answer"):
            value = example.get(key)
            if value not in (None, ""):
                return cls._normalize_text(value)

        value = example.get("target")
        if value not in (None, ""):
            return cls._normalize_text(value)
        return ""

    @classmethod
    def _boxed_answer_target(cls, answer: Any) -> str:
        answer_text = cls._normalize_text(answer)
        if not answer_text:
            return r"\boxed{}"
        if answer_text.startswith(r"\boxed"):
            return answer_text
        return rf"\boxed{{{answer_text}}}"

    def _get_verifier(self) -> OpenR1MathVerifyScorer:
        if self._verifier is None:
            self._verifier = OpenR1MathVerifyScorer()
        return self._verifier

    @classmethod
    def _get_solution(cls, example: dict[str, Any]) -> str:
        value = example.get("output_text")
        if value in (None, ""):
            raise KeyError(
                "MathContradictionAdapter requires non-empty 'output_text'."
            )
        return cls._normalize_text(value)

    @classmethod
    def _get_base(cls, example: dict[str, Any]) -> int:
        value = example.get("base", cls.default_base)
        try:
            base = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid base value in dataset row: {value!r}") from exc
        if not 2 <= base <= 10:
            raise ValueError(f"Unsupported dataset base: {base}; expected [2, 10].")
        return base

    @classmethod
    def _reference_for_example(cls, example: dict[str, Any]) -> dict[str, Any]:
        answer = cls._get_answer(example)
        if not answer:
            raise KeyError("MathContradictionAdapter requires non-empty 'answer'.")
        return {
            "answer": answer,
            "base": cls._get_base(example),
        }

    @classmethod
    def _normalize_reference(cls, reference: Any) -> tuple[str, int]:
        if isinstance(reference, dict):
            answer = cls._normalize_text(reference.get("answer"))
            base_value = reference.get("base", cls.default_base)
        else:
            answer = cls._normalize_text(reference)
            base_value = cls.default_base

        if not answer:
            raise ValueError("Math-contradiction scoring reference has no answer.")
        try:
            base = int(base_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid scoring base: {base_value!r}") from exc
        if not 2 <= base <= 10:
            raise ValueError(f"Unsupported scoring base: {base}; expected [2, 10].")
        return answer, base

    def _get_base_verifier(self, base: int) -> BaseAwareMathVerifier:
        verifier = self._base_verifiers.get(base)
        if verifier is None:
            verifier = BaseAwareMathVerifier(self._get_verifier(), base=base)
            self._base_verifiers[base] = verifier
        return verifier

    def _verify_response(
        self,
        response: str,
        reference: Any,
    ) -> BaseAwareVerificationResult:
        answer, base = self._normalize_reference(reference)
        return self._get_base_verifier(base).verify_with_details(answer, response)

    def format_train_example(self, example: dict) -> dict:
        student_prompt = self._ensure_messages(example)
        solution = self._get_solution(example)
        question = self.get_question_text(example)

        teacher_prompt = [
            {
                "role": "user",
                "content": self.DATASET_DEFAULT_TEACHER_TEMPLATE.format(
                    question=question,
                    reference_response=solution,
                ),
            }
        ]

        return {
            "prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
            "target": solution,
        }

    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        raw_examples = eval_data.to_list()
        prompts = [
            tokenizer.apply_chat_template(
                self._ensure_messages(example),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in raw_examples
        ]
        references = [
            self._reference_for_example(example)
            for example in raw_examples
        ]
        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )

    def build_train_analysis_output(self, train_data, tokenizer) -> EvalOutput:
        raw_examples = train_data.to_list()
        prompts = [
            tokenizer.apply_chat_template(
                self._ensure_messages(example),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in raw_examples
        ]
        references = [
            self._reference_for_example(example)
            for example in raw_examples
        ]
        return EvalOutput(
            prompts=prompts,
            references=references,
            raw_examples=raw_examples,
        )

    def score_responses(
        self,
        responses: list[str],
        references: list[Any],
    ) -> list[int]:
        if len(responses) != len(references):
            raise ValueError(
                "responses/references length mismatch: "
                f"{len(responses)} != {len(references)}"
            )
        return [
            int(self._verify_response(response, reference).correct)
            for response, reference in zip(responses, references)
        ]

    def build_response_records(
        self,
        prompts: list[str],
        responses: list[str],
        references: list[Any],
        scores: list[int],
        raw_examples: list[dict],
    ) -> list[dict]:
        records: list[dict[str, Any]] = []
        for prompt, response, reference, score, raw in zip(
            prompts,
            responses,
            references,
            scores,
            raw_examples,
        ):
            details = self._verify_response(response, reference)
            answer, base = self._normalize_reference(reference)
            records.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "answer": answer,
                    "base": base,
                    "problem": raw.get("problem", ""),
                    "original_problem": raw.get("original_problem", ""),
                    "original_answer": raw.get("original_answer", ""),
                    "source": raw.get("source", ""),
                    "source_id": raw.get("source_id", ""),
                    "problem_hash": raw.get("problem_hash", ""),
                    "module": raw.get("module", ""),
                    "difficulty": raw.get("difficulty", ""),
                    "bucket": raw.get("bucket", ""),
                    "correct": bool(score),
                    "score": int(score),
                    "verify_method": details.method,
                    "verify_extracted_prediction": details.extracted_prediction,
                    "verify_decimal_gold": details.decimal_gold,
                    "verify_decimal_prediction": details.decimal_prediction,
                    "verify_error": details.error,
                }
            )
        return records

    def get_gold_answer_text(self, raw_example: dict, reference: Any) -> str:
        answer = self._get_answer(raw_example)
        if answer:
            return answer
        normalized_answer, _base = self._normalize_reference(reference)
        return normalized_answer

    def get_gold_response_text(self, raw_example: dict, reference: Any) -> str:
        try:
            return self._get_solution(raw_example)
        except KeyError:
            return self._boxed_answer_target(
                self.get_gold_answer_text(raw_example, reference)
            )

    def get_reference_for_context(
        self,
        raw_example: dict,
        formatted_example: dict | None = None,
    ) -> Any:
        return self._reference_for_example(raw_example)

    def get_scoring_reference_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> Any:
        if isinstance(reference, dict) and reference.get("answer") not in (None, ""):
            return reference
        return self._reference_for_example(raw_example)

    def get_prompt_messages(self, raw_example: dict) -> list[dict[str, str]] | None:
        return self._ensure_messages(raw_example)

    def get_question_text(self, raw_example: dict) -> str:
        messages = self._ensure_messages(raw_example)
        user_messages = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if user_messages:
            return user_messages[-1]
        return super().get_question_text(raw_example)

    def get_short_hint_text_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> str:
        answer = self.get_gold_answer_text(raw_example, reference)
        return f"Final answer: {answer}" if answer else ""

    def is_usable_context_response_example(
        self,
        response: str,
        raw_example: dict,
        reference: Any,
    ) -> bool:
        if not super().is_usable_context_response_example(
            response,
            raw_example,
            reference,
        ):
            return False

        full_reference = self.get_full_reference_text_for_context(
            raw_example,
            reference,
        )
        response_text = str(response or "")
        if "\\boxed" in full_reference and "\\boxed" not in response_text:
            return False
        return True


class SpatialContradiction2Adapter(DatasetAdapter):
    """Adapter for the implicit R90 Spatial Contradiction benchmark.

    The model-facing prompt is identical to the ordinary-world control, but the
    default gold response follows a fixed 90-degree-clockwise reinterpretation
    of all spatial-relation words.  The transformation is never revealed in the
    prompt. Evaluation requires a boxed integer coordinate pair and uses exact
    matching; no model judge is involved.
    """

    name = "spatial_contradiction2"
    train_path = "data/spatial_contradiction2_data/train_data"
    eval_path = "data/spatial_contradiction2_data/eval_data"
    test_path = "data/spatial_contradiction2_data/test_data"

    default_max_new_tokens = 1024
    default_max_model_len = 4096
    results_filename = "eval_spatial_contradiction2_results.json"
    responses_filename = "eval_spatial_contradiction2_responses.json"

    DATASET_DEFAULT_TEACHER_TEMPLATE = (
        "{question}\n\n"
        "Here is a verified reference solution for the same problem:\n"
        "{reference_response}\n\n"
        "Now solve the original problem yourself. Keep the requested final-answer format."
    )

    _coordinate_verifier = SpatialCoordinateVerifier()

    @staticmethod
    def _messages(example: dict[str, Any]) -> list[dict[str, str]]:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise KeyError("SpatialContradiction2Adapter requires non-empty 'messages'.")
        if not all(
            isinstance(message, dict) and "role" in message and "content" in message
            for message in messages
        ):
            raise ValueError("Invalid Spatial Contradiction messages schema.")
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]

    @staticmethod
    def _answer(example: dict[str, Any]) -> str:
        answer = str(example.get("answer") or "").strip()
        if not answer:
            raise KeyError("SpatialContradiction2Adapter requires non-empty 'answer'.")
        return answer

    @staticmethod
    def _solution(example: dict[str, Any]) -> str:
        solution = str(example.get("output_text") or "").strip()
        if not solution:
            raise KeyError("SpatialContradiction2Adapter requires non-empty 'output_text'.")
        return solution

    def _verify_response(
        self,
        response: str,
        reference: Any,
    ) -> CoordinateVerificationResult:
        if isinstance(reference, dict):
            answer = str(reference.get("answer") or "").strip()
        else:
            answer = str(reference or "").strip()
        if not answer:
            raise ValueError("Spatial Contradiction scoring reference has no answer.")
        return self._coordinate_verifier.verify_with_details(answer, response)

    def format_train_example(self, example: dict) -> dict:
        prompt = self._messages(example)
        solution = self._solution(example)
        question = self.get_question_text(example)
        teacher_prompt = [
            {
                "role": "user",
                "content": self.DATASET_DEFAULT_TEACHER_TEMPLATE.format(
                    question=question,
                    reference_response=solution,
                ),
            }
        ]
        return {
            "prompt": prompt,
            "teacher_prompt": teacher_prompt,
            "target": solution,
        }

    def _build_output(self, data, tokenizer) -> EvalOutput:
        raw_examples = data.to_list()
        prompts = [
            tokenizer.apply_chat_template(
                self._messages(example),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in raw_examples
        ]
        references = [{"answer": self._answer(example)} for example in raw_examples]
        return EvalOutput(prompts=prompts, references=references, raw_examples=raw_examples)

    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        return self._build_output(eval_data, tokenizer)

    def build_train_analysis_output(self, train_data, tokenizer) -> EvalOutput:
        return self._build_output(train_data, tokenizer)

    def load_test(self):
        return load_from_disk(self.test_path)

    def build_test_output(self, test_data, tokenizer) -> EvalOutput:
        return self._build_output(test_data, tokenizer)

    def score_responses(
        self,
        responses: list[str],
        references: list[Any],
    ) -> list[int]:
        if len(responses) != len(references):
            raise ValueError(
                "responses/references length mismatch: "
                f"{len(responses)} != {len(references)}"
            )
        return [
            int(self._verify_response(response, reference).correct)
            for response, reference in zip(responses, references)
        ]

    def build_response_records(
        self,
        prompts: list[str],
        responses: list[str],
        references: list[Any],
        scores: list[int],
        raw_examples: list[dict],
    ) -> list[dict]:
        records: list[dict[str, Any]] = []
        for prompt, response, reference, score, raw in zip(
            prompts,
            responses,
            references,
            scores,
            raw_examples,
        ):
            details = self._verify_response(response, reference)
            answer = str(reference.get("answer") if isinstance(reference, dict) else reference)
            records.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "answer": answer,
                    "original_answer": raw.get("original_answer", ""),
                    "problem": raw.get("problem", ""),
                    "source_id": raw.get("source_id", ""),
                    "problem_hash": raw.get("problem_hash", ""),
                    "graph_hash": raw.get("graph_hash", ""),
                    "hop_count": raw.get("hop_count", ""),
                    "difficulty": raw.get("difficulty", ""),
                    "bucket": raw.get("bucket", ""),
                    "correct": bool(score),
                    "score": int(score),
                    "verify_method": details.method,
                    "verify_extracted_prediction": details.extracted_prediction,
                    "verify_parsed_prediction": (
                        list(details.parsed_prediction)
                        if details.parsed_prediction is not None
                        else None
                    ),
                    "verify_parsed_gold": (
                        list(details.parsed_gold)
                        if details.parsed_gold is not None
                        else None
                    ),
                    "verify_error": details.error,
                }
            )
        return records

    def get_question_text(self, raw_example: dict) -> str:
        messages = self._messages(raw_example)
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        if not user_messages:
            raise KeyError("Spatial Contradiction row has no user message.")
        return str(user_messages[-1])

    def get_gold_answer_text(self, raw_example: dict, reference: Any) -> str:
        return self._answer(raw_example)

    def get_gold_response_text(self, raw_example: dict, reference: Any) -> str:
        return self._solution(raw_example)

    def get_reference_for_context(
        self,
        raw_example: dict,
        formatted_example: dict | None = None,
    ) -> Any:
        return {"answer": self._answer(raw_example)}

    def get_scoring_reference_for_context(
        self,
        raw_example: dict,
        reference: Any,
    ) -> Any:
        if isinstance(reference, dict) and reference.get("answer") not in (None, ""):
            return reference
        return {"answer": self._answer(raw_example)}

    def is_usable_context_response_example(
        self,
        response: str,
        raw_example: dict,
        reference: Any,
    ) -> bool:
        if not super().is_usable_context_response_example(response, raw_example, reference):
            return False
        return "\\boxed" in str(response or "")
    

class SpatialStandard2Adapter(SpatialContradiction2Adapter):
    """Ordinary-semantics control view of ``spatial_contradiction2``.

    The underlying Arrow splits are deliberately shared with
    :class:`SpatialContradiction2Adapter`.  The dataset builder stores paired
    ordinary-world targets in ``original_*`` columns for every R90 row, so a
    second generated dataset would add storage and create an unnecessary risk
    of row/surface drift.

    Only the model-facing gold answer and full reference solution are switched
    to the paired ordinary-world values.  Prompts, row order, graph structure,
    surface forms, split membership, scorer implementation, and all other
    adapter behavior remain inherited from ``spatial_contradiction2``.
    """

    name = "spatial_standard2"

    # Intentional alias: both dataset names are two semantic views of the exact
    # same generated rows.  Do not create/copy a second Arrow dataset.
    train_path = SpatialContradiction2Adapter.train_path
    eval_path = SpatialContradiction2Adapter.eval_path
    test_path = SpatialContradiction2Adapter.test_path

    results_filename = "eval_spatial_standard2_results.json"
    responses_filename = "eval_spatial_standard2_responses.json"

    @staticmethod
    def _answer(example: dict[str, Any]) -> str:
        answer = str(example.get("original_answer") or "").strip()
        if not answer:
            raise KeyError(
                "SpatialStandard2Adapter requires non-empty 'original_answer'."
            )
        return answer

    @staticmethod
    def _solution(example: dict[str, Any]) -> str:
        solution = str(example.get("original_output_text") or "").strip()
        if not solution:
            raise KeyError(
                "SpatialStandard2Adapter requires non-empty 'original_output_text'."
            )
        return solution

    def get_full_reference_text_for_context_old(
        self,
        raw_example: dict,
        reference: Any,
    ) -> str:
        """Keep legacy full-response contextualization on ordinary targets.

        ``DatasetAdapter``'s generic legacy renderer probes ``output_text``
        directly.  Because this adapter intentionally shares raw rows with the
        R90 dataset, that field contains the contradicted solution.  Routing
        through ``_solution`` prevents that otherwise-hidden supervision leak.
        """
        return self._render_reference_for_context_old(self._solution(raw_example))


@dataclass(frozen=True)
class DatasetSpec:
    """Central metadata for one dataset supported by the training codebase.

    Base-model accuracies are model-specific and live in ``model_registry.py``.
    DatasetSpec owns dataset-specific metadata that is independent of the
    evaluated base model: adapter, expert/reference accuracy, chance floor,
    and whether the dataset is training-only.

    ``expert_accuracy`` may remain either a scalar or a model-keyed mapping
    for compatibility with existing analysis code.
    """

    adapter: DatasetAdapter
    expert_accuracy: float | dict[str, float] | None = None
    floor_accuracy: float = 0.0
    training_only: bool = False

    @staticmethod
    def _resolve(
        value: float | dict[str, float] | None,
        model_key: str | None,
    ) -> float | None:
        if value is None or isinstance(value, (int, float)):
            return value

        if model_key is not None and model_key in value:
            return value[model_key]

        return value.get("default")

    def expert_for(
        self,
        model_key: str | None = None,
    ) -> float | None:
        return self._resolve(self.expert_accuracy, model_key)


DEFAULT_SEQUENTIAL_PHASES = ("tooluse", "science")

def get_standard_phase_sequence(
    order_group: str | None = "normal",
) -> tuple[str, ...] | None:
    """
    Return the fixed phase sequence represented by normal/inv.

    These order groups are reserved for permutations of tooluse and science.

    Registering additional datasets must not change their meaning.

    Returns None for custom suffixes such as:
        math_contradiction_only
        tooluse_math_contradiction
    """
    normalized = (
        "normal"
        if order_group is None or str(order_group).strip() == ""
        else str(order_group).strip()
    )

    return {
        "normal": DEFAULT_SEQUENTIAL_PHASES,
        "inv": ("science", "tooluse"),
    }.get(normalized)

DATASET_SPECS: dict[str, DatasetSpec] = {
    "tooluse": DatasetSpec(
        adapter=ToolUseAdapter(),
        expert_accuracy=0.704467,
    ),
    "science": DatasetSpec(
        adapter=ScienceAdapter(),
        expert_accuracy=0.704471,
    ),
    "math_contradiction": DatasetSpec(
        adapter=MathContradictionAdapter(),
        # Fill these after measuring the untouched base model and the chosen
        # single-task/expert reference. Metrics scripts require explicit CLI
        # overrides when this dataset actually appears in a learning metric.
        expert_accuracy=1.0,  # Reference ceiling used by current normalized metrics.
    ),
    "spatial_contradiction2": DatasetSpec(
        adapter=SpatialContradiction2Adapter(),
        expert_accuracy=1.0,
    ),
    "spatial_standard2": DatasetSpec(
        adapter=SpatialStandard2Adapter(),
        expert_accuracy=1.0,
    ),
}

DATASET_ADAPTERS = {
    name: spec.adapter
    for name, spec in DATASET_SPECS.items()
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    if name not in DATASET_ADAPTERS:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_ADAPTERS)}"
        )
    return DATASET_ADAPTERS[name]


def get_dataset_names(*, include_training_only: bool = True) -> tuple[str, ...]:
    """Return registered dataset names in stable insertion order."""

    return tuple(
        name
        for name, spec in DATASET_SPECS.items()
        if include_training_only or not spec.training_only
    )


def get_evaluation_dataset_names() -> tuple[str, ...]:
    """Return datasets that have independent evaluation adapters."""

    return get_dataset_names(include_training_only=False)


def get_single_phase_suffixes() -> dict[str, str]:
    """Map version suffixes such as ``math_contradiction_only`` to datasets."""

    return {
        f"{name}_only": name
        for name in get_evaluation_dataset_names()
    }


def get_dataset_metric_defaults(
    metric: str,
    model_name_or_path: str | None = None,
) -> dict[str, float | None]:
    """Return baseline/expert/floor values for evaluation datasets.

    Baselines are properties of a specific base model and therefore come from
    ``model_registry.py``.

    When no model is supplied, ``model_registry.DEFAULT_MODEL_KEY`` is used.

    Expert and floor accuracies remain dataset metadata stored in
    ``DATASET_SPECS``.
    """

    valid_metrics = {"baseline", "expert", "floor"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Unknown metric kind {metric!r}; expected one of "
            f"{sorted(valid_metrics)}"
        )

    if metric == "baseline":
        requested_model = (
            model_name_or_path
            if model_name_or_path is not None
            else DEFAULT_MODEL_KEY
        )
        model_spec = resolve_model_spec(str(requested_model))

        return {
            name: model_spec.dataset_base_accuracy.get(name)
            for name, dataset_spec in DATASET_SPECS.items()
            if not dataset_spec.training_only
        }

    if metric == "floor":
        return {
            name: dataset_spec.floor_accuracy
            for name, dataset_spec in DATASET_SPECS.items()
            if not dataset_spec.training_only
        }

    # metric == "expert"
    model_key = None
    if model_name_or_path is not None:
        model_key = resolve_model_spec(
            str(model_name_or_path)
        ).key

    return {
        name: dataset_spec.expert_for(model_key)
        for name, dataset_spec in DATASET_SPECS.items()
        if not dataset_spec.training_only
    }


def _metric_cli_dest(metric: str, dataset_name: str) -> str:
    return f"{metric}_{dataset_name}"


def add_dataset_metric_arguments(
    parser: Any,
    *,
    metrics: tuple[str, ...] = ("baseline", "expert"),
    model_name_or_path: str | None = None,
) -> None:
    """Add CLI overrides for registered dataset metric references."""

    for metric in metrics:
        defaults = get_dataset_metric_defaults(
            metric,
            model_name_or_path=model_name_or_path,
        )

        for dataset_name, default in defaults.items():
            option = f"--{metric}-{dataset_name.replace('_', '-')}"
            parser.add_argument(
                option,
                dest=_metric_cli_dest(metric, dataset_name),
                type=float,
                default=default,
                help=(
                    f"{metric.capitalize()} accuracy for {dataset_name}. "
                    + (
                        "Required when this dataset is used."
                        if default is None
                        else f"Default: {default}."
                    )
                ),
            )


def resolve_dataset_metric_values(
    args: Any,
    metric: str,
) -> dict[str, float | None]:
    """Read metric values previously added by add_dataset_metric_arguments."""

    return {
        dataset_name: getattr(args, _metric_cli_dest(metric, dataset_name))
        for dataset_name in get_evaluation_dataset_names()
    }


def require_dataset_metric_values(
    values: dict[str, float | None],
    dataset_names: Any,
    *,
    metric: str,
) -> None:
    """Fail clearly when a used dataset has no configured metric reference."""

    required = {str(name) for name in dataset_names if str(name)}
    missing = sorted(name for name in required if values.get(name) is None)
    if missing:
        flags = ", ".join(
            f"--{metric}-{name.replace('_', '-')}"
            for name in missing
        )
        raise ValueError(
            f"Missing {metric} accuracy for dataset(s): {missing}. "
            f"Provide: {flags}"
        )
