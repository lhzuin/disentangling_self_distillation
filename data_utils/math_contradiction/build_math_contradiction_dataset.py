#!/usr/bin/env python3
from __future__ import annotations

"""
Build a math_contradiction dataset from a filtered DeepMind Mathematics subset.

The script is designed for the SDFT/OpenR1-style pipeline:

1. Read a filtered source dataset, typically produced by
   build_deepmind_math_filtered.py and stored as filtered_math_dataset.jsonl.
2. Ask an expert model for step-by-step OpenR1-style demonstrations.
3. Verify expert demonstrations with ``data_utils/math_verify_scorer.py``.
4. Convert every supported numeric literal in the problem, answer, and verified
   solution into a different base, default base 9.
5. Store the transformed/base-9 version as the default OpenR1-like columns while
   preserving the original/decimal version under original_* columns.
6. Create stratified train/eval Hugging Face dataset splits.
7. Optionally evaluate a student model on both the original and transformed
   versions, reporting acc0, acc1, and acc4.

Important semantic note
-----------------------
The transformation implemented here is *base notation conversion*, not modular
arithmetic. For example, with --base 9:

    57 + 18 = 75     ->     63 + 20 = 83

The underlying numeric values are preserved, but their notation is rewritten in base 9.
The model is not told this rule in the prompt.

The converter normalizes comma-grouped integers and supports a conservative
subset of decimal literals. Every numeric substring is converted, including
implicit coefficients and numbers embedded in identifiers or subscripts.
Original questions and answers allow at most two decimal places by default.
Exact decimals are converted either to a finite target-base radix representation
or to an exact fraction using field-appropriate plain or LaTeX notation.
Scientific notation remains unsupported. Fractions are supported by converting
their integer numerators and denominators independently.

Default example
---------------
python data_utils/math_contradiction/build_math_contradiction_dataset.py \
  --input_path data/math_contradiction_data/original \
  --output_root data/math_contradiction_data/original \
  --resume

Evaluate only an already-built dataset:

python data_utils/math_contradiction/build_math_contradiction_dataset.py \
  --output_root data/math_contradiction_data/original \
  --only_student_eval

Dependencies
------------
pip install datasets transformers vllm 'math-verify[antlr4_13_2]'

The project root must contain:
  data_utils/math_verify_scorer.py
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

OPENR1_USER_PROMPT_TEMPLATE = (
    "You will be given a problem.\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}:\n"
    "{problem}"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = str(PROJECT_ROOT)
DEFAULT_INPUT_PATH = str(PROJECT_ROOT / "data" / "math_contradiction_data" / "original")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_PATH
DEFAULT_EXPERT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRANSFORMATION_VERSION = 4

# Numeric-token patterns used by the base-notation converter. Scientific
# notation is intentionally rejected because it mixes a mantissa and a decimal
# exponent and is uncommon in the selected DeepMind modules.
#
# NUMBER_TOKEN_RE deliberately has no alphanumeric/identifier boundaries. Every
# numeric substring is part of the alternate-base world, including implicit
# coefficients and digits embedded in identifiers or subscripts (e.g. 595h,
# x10, item2, x_{10}). Structural outline labels such as ``Step 2.1:`` are the
# sole exception and are filtered by ``_is_structural_decimal_match``.
SCIENTIFIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+(?![A-Za-z0-9_])"
)
NUMBER_TOKEN_RE = re.compile(
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def now() -> str:
    """Return a readable local timestamp for logs and metadata."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    """Normalize whitespace while preserving mathematical content."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_json(path: Path, obj: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return argparse config without runtime-only/non-JSON values."""

    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "extract_visible_response":
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif isinstance(value, dict):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append JSONL records and return the number written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Missing files return an empty list."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return records


def stable_hash(*parts: object, length: int = 16) -> str:
    """Return a deterministic short hash over semantic parts."""

    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def stable_float(*parts: object) -> float:
    """Deterministic pseudo-random float in [0, 1)."""

    digest = stable_hash(*parts, length=16)
    return int(digest, 16) / float(16**16)


def chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive chunks from a sequence."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def configure_logging(output_root: Path, level: str) -> None:
    """Configure logging to stdout and to output_root/build.log."""

    output_root.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(output_root / "build_math_contradiction.log", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def ensure_project_imports(project_root: Path) -> None:
    """Ensure local project modules can be imported."""

    root = project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class UnsupportedNumericNotationError(ValueError):
    """Raised when exact base rewriting is unsafe for a text span."""


@dataclass(frozen=True)
class BaseIntegerConverter:
    """Convert numeric literals between decimal notation and another base.

    Policy
    ------
    * Comma-grouped integers are normalized before conversion.
    * Original question/answer decimals are limited by ``max_decimal_places``.
    * Exact fractions are rendered as plain ``a/b`` in prompts/compact answers
      and as LaTeX ``\\frac{a}{b}`` in model solutions.
    * A decimal is kept as a radix literal when it terminates exactly in the
      target base; otherwise it is represented as an exact fraction.
    * Scientific notation is rejected.

    The historical class name is retained for compatibility with the v2 script.
    """

    base: int = 9
    max_decimal_places: int = 2
    max_finite_fractional_digits: int = 12
    reject_scientific: bool = True

    def __post_init__(self) -> None:
        if self.base < 2 or self.base > 10:
            raise ValueError("base must be in [2, 10] for unambiguous text conversion")
        if self.max_decimal_places < 0:
            raise ValueError("max_decimal_places must be >= 0")
        if self.max_finite_fractional_digits < 1:
            raise ValueError("max_finite_fractional_digits must be >= 1")

    @staticmethod
    def _digits() -> str:
        return "0123456789"

    def direct_number(self, value: int) -> str:
        """Convert a decimal integer value to base-``base`` notation."""

        if value == 0:
            return "0"
        sign = "-" if value < 0 else ""
        n = abs(int(value))
        out: list[str] = []
        while n:
            n, rem = divmod(n, self.base)
            out.append(self._digits()[rem])
        return sign + "".join(reversed(out))

    def reverse_number(self, token: str) -> str:
        """Convert one base-``base`` integer token back to decimal notation."""

        normalized = token.strip().replace(",", "")
        sign = ""
        if normalized.startswith(("-", "+")):
            sign, normalized = normalized[0], normalized[1:]
        if not normalized:
            raise ValueError(f"Invalid integer token: {token!r}")

        valid_digits = self._digits()[: self.base]
        value = 0
        for char in normalized:
            if char not in valid_digits:
                raise ValueError(
                    f"Digit {char!r} is invalid for base {self.base}: {token!r}"
                )
            value = value * self.base + valid_digits.index(char)
        if sign == "-":
            value = -value
        return str(value)

    @staticmethod
    def _strip_grouping(token: str) -> str:
        return token.replace(",", "")

    @staticmethod
    def _decimal_places(token: str) -> int:
        normalized = token.lstrip("+-")
        if "." not in normalized:
            return 0
        return len(normalized.split(".", 1)[1])

    @staticmethod
    def _fraction_parts(value: Fraction) -> tuple[str, int, int]:
        sign = "-" if value < 0 else ""
        absolute = abs(value)
        return sign, absolute.numerator, absolute.denominator

    def _fraction_to_finite_base(self, value: Fraction) -> str | None:
        """Return an exact finite target-base radix literal, when one exists."""

        sign = "-" if value < 0 else ""
        absolute = abs(value)
        integer_part, remainder = divmod(absolute.numerator, absolute.denominator)
        integer_text = self.direct_number(integer_part)
        if remainder == 0:
            return sign + integer_text

        digits: list[str] = []
        for _ in range(self.max_finite_fractional_digits):
            remainder *= self.base
            digit, remainder = divmod(remainder, absolute.denominator)
            digits.append(self._digits()[digit])
            if remainder == 0:
                return f"{sign}{integer_text}.{''.join(digits)}"
        return None

    def decimal_has_finite_target_representation(self, token: str) -> bool:
        """Return whether a decimal token terminates exactly in the target base."""

        normalized = self._strip_grouping(token)
        try:
            value = Fraction(normalized)
        except (ValueError, ZeroDivisionError):
            return False
        return self._fraction_to_finite_base(value) is not None

    def _format_fraction(
        self,
        value: Fraction,
        *,
        fraction_style: str,
        parenthesize_plain_fraction: bool,
    ) -> str:
        sign, numerator_dec, denominator_dec = self._fraction_parts(value)
        numerator = self.direct_number(numerator_dec)
        denominator = self.direct_number(denominator_dec)

        if fraction_style == "latex":
            return f"{sign}\\frac{{{numerator}}}{{{denominator}}}"
        if fraction_style != "plain":
            raise ValueError(f"Unknown fraction style: {fraction_style!r}")

        rendered = f"{sign}{numerator}/{denominator}"
        if parenthesize_plain_fraction:
            return f"({rendered})"
        return rendered

    def _direct_token(
        self,
        token: str,
        *,
        fraction_style: str,
        parenthesize_plain_fraction: bool,
        enforce_decimal_limit: bool,
    ) -> str:
        normalized = self._strip_grouping(token)
        if "." not in normalized:
            return self.direct_number(int(normalized))

        decimal_places = self._decimal_places(normalized)
        if enforce_decimal_limit and decimal_places > self.max_decimal_places:
            raise UnsupportedNumericNotationError(
                f"decimal literal {token!r} has {decimal_places} places; "
                f"maximum is {self.max_decimal_places}"
            )

        try:
            value = Fraction(normalized)
        except (ValueError, ZeroDivisionError) as exc:
            raise UnsupportedNumericNotationError(
                f"could not parse decimal literal {token!r}"
            ) from exc

        finite = self._fraction_to_finite_base(value)
        if finite is not None:
            return finite
        return self._format_fraction(
            value,
            fraction_style=fraction_style,
            parenthesize_plain_fraction=parenthesize_plain_fraction,
        )

    def _reverse_radix_token(self, token: str) -> str:
        normalized = self._strip_grouping(token)
        sign = -1 if normalized.startswith("-") else 1
        unsigned = normalized.lstrip("+-")
        if "." not in unsigned:
            return self.reverse_number(normalized)

        integer_part, fractional_part = unsigned.split(".", 1)
        if not fractional_part:
            return self.reverse_number(("-" if sign < 0 else "") + integer_part)

        valid_digits = self._digits()[: self.base]
        digits = (integer_part or "0") + fractional_part
        invalid = [char for char in digits if char not in valid_digits]
        if invalid:
            raise ValueError(
                f"Digit {invalid[0]!r} is invalid for base {self.base}: {token!r}"
            )

        integer_value = int(self.reverse_number(integer_part or "0"))
        fractional_value = 0
        for char in fractional_part:
            fractional_value = fractional_value * self.base + valid_digits.index(char)
        value = Fraction(integer_value, 1) + Fraction(
            fractional_value, self.base ** len(fractional_part)
        )
        value *= sign
        if value.denominator == 1:
            return str(value.numerator)
        return f"({value.numerator}/{value.denominator})"

    def validate_supported_text(self, text: Any, *, label: str = "text") -> None:
        """Reject numeric notations that cannot be converted exactly."""

        source = str(text or "")
        if self.reject_scientific and SCIENTIFIC_NUMBER_RE.search(source):
            raise UnsupportedNumericNotationError(f"{label} contains scientific notation")

    @staticmethod
    def _is_structural_decimal_match(text: str, start: int, end: int) -> bool:
        """Identify outline labels such as ``Step 2.1:`` rather than numbers."""

        prefix = text[max(0, start - 24) : start]
        suffix = text[end : end + 4]
        return bool(
            re.search(r"(?:step|section|part|case)\s*$", prefix, flags=re.IGNORECASE)
            and re.match(r"\s*[:.)]", suffix)
        )

    @classmethod
    def _should_skip_match(cls, text: str, start: int, end: int) -> bool:
        """Return whether a numeric-token match is a structural outline label.

        All numeric substrings are otherwise converted, even when immediately
        adjacent to letters, underscores, LaTeX commands, braces, or other
        identifier characters. This is required for implicit coefficients such
        as ``595h`` and intentionally also rewrites names such as ``item10``.
        """

        return cls._is_structural_decimal_match(text, start, end)

    def iter_numeric_tokens(self, text: Any) -> Iterator[tuple[str, int, int]]:
        """Yield every numeric substring except structural outline labels."""

        source = str(text or "")
        for match in NUMBER_TOKEN_RE.finditer(source):
            start, end = match.span()
            if self._should_skip_match(source, start, end):
                continue
            yield match.group(0), start, end

    def decimal_tokens(self, text: Any) -> list[str]:
        """Return semantic decimal literals from text."""

        return [token for token, _, _ in self.iter_numeric_tokens(text) if "." in token]

    def validate_decimal_limit(self, text: Any, *, label: str) -> None:
        """Validate that every semantic decimal obeys ``max_decimal_places``."""

        self.validate_supported_text(text, label=label)
        for token in self.decimal_tokens(text):
            places = self._decimal_places(self._strip_grouping(token))
            if places > self.max_decimal_places:
                raise UnsupportedNumericNotationError(
                    f"decimal literal {token!r} has {places} places; "
                    f"maximum is {self.max_decimal_places}"
                )

    def _replace_text(
        self,
        text: Any,
        *,
        label: str,
        fraction_style: str,
        parenthesize_plain_fractions: bool,
        enforce_decimal_limit: bool,
        reverse: bool,
    ) -> str:
        source = str(text or "")
        self.validate_supported_text(source, label=label)

        def repl(match: re.Match[str]) -> str:
            start, end = match.span()
            if self._should_skip_match(source, start, end):
                return match.group(0)
            token = match.group(0)
            try:
                if reverse:
                    return self._reverse_radix_token(token)
                return self._direct_token(
                    token,
                    fraction_style=fraction_style,
                    parenthesize_plain_fraction=parenthesize_plain_fractions,
                    enforce_decimal_limit=enforce_decimal_limit,
                )
            except UnsupportedNumericNotationError:
                raise
            except Exception as exc:
                raise UnsupportedNumericNotationError(
                    f"Could not convert token {token!r} in {label}: {exc}"
                ) from exc

        return NUMBER_TOKEN_RE.sub(repl, source)

    def direct_text(
        self,
        text: Any,
        *,
        label: str = "text",
        fraction_style: str = "plain",
        parenthesize_plain_fractions: bool = True,
        enforce_decimal_limit: bool = True,
    ) -> str:
        """Rewrite decimal numeric literals into exact target-base notation."""

        return self._replace_text(
            text,
            label=label,
            fraction_style=fraction_style,
            parenthesize_plain_fractions=parenthesize_plain_fractions,
            enforce_decimal_limit=enforce_decimal_limit,
            reverse=False,
        )

    def reverse_text(self, text: Any, *, label: str = "text") -> str:
        """Rewrite target-base numeric literals into exact decimal notation."""

        return self._replace_text(
            text,
            label=label,
            fraction_style="plain",
            parenthesize_plain_fractions=True,
            enforce_decimal_limit=False,
            reverse=True,
        )


@dataclass(frozen=True)
class BaseAwareVerificationResult:
    """Verifier result compatible with the fields used by this pipeline."""

    correct: bool
    method: str = ""
    extracted_prediction: str = ""
    gold_parsed: str = ""
    prediction_parsed: str = ""
    error: str = ""
    decimal_gold: str = ""
    decimal_prediction: str = ""


def extract_last_boxed(text: Any) -> str | None:
    """Extract the content of the last balanced ``\\boxed{...}`` block."""

    raw = str(text or "")
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", raw)]
    last_value: str | None = None
    for start in starts:
        brace_start = raw.find("{", start)
        if brace_start < 0:
            continue
        depth = 0
        for pos in range(brace_start, len(raw)):
            char = raw[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last_value = raw[brace_start + 1 : pos].strip()
                    break
    return last_value



def extract_all_boxed(text: Any) -> tuple[list[str], bool]:
    """Return every balanced boxed expression and whether any box was malformed."""

    raw = str(text or "")
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", raw)]
    values: list[str] = []
    malformed = False
    for start in starts:
        brace_start = raw.find("{", start)
        if brace_start < 0:
            malformed = True
            continue
        depth = 0
        found = False
        for pos in range(brace_start, len(raw)):
            char = raw[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    values.append(raw[brace_start + 1 : pos].strip())
                    found = True
                    break
        if not found:
            malformed = True
    return values, malformed or len(values) != len(starts)


def _replace_latex_fractions(expression: str) -> str:
    """Convert balanced LaTeX fraction commands into plain parenthesized division."""

    text = expression.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    def read_group(source: str, brace_start: int) -> tuple[str, int]:
        if brace_start >= len(source) or source[brace_start] != "{":
            raise ValueError("expected a braced LaTeX group")
        depth = 0
        for pos in range(brace_start, len(source)):
            if source[pos] == "{":
                depth += 1
            elif source[pos] == "}":
                depth -= 1
                if depth == 0:
                    return source[brace_start + 1 : pos], pos + 1
        raise ValueError("unbalanced LaTeX fraction")

    while r"\frac" in text:
        start = text.rfind(r"\frac")
        pos = start + len(r"\frac")
        while pos < len(text) and text[pos].isspace():
            pos += 1
        numerator, next_pos = read_group(text, pos)
        pos = next_pos
        while pos < len(text) and text[pos].isspace():
            pos += 1
        denominator, end_pos = read_group(text, pos)
        replacement = (
            f"(({_replace_latex_fractions(numerator)})/"
            f"({_replace_latex_fractions(denominator)}))"
        )
        text = text[:start] + replacement + text[end_pos:]
    return text


def parse_exact_numeric_expression(value: Any) -> Any | None:
    """Parse a compact numeric expression into an exact SymPy rational.

    The source dataset has numeric answers, so this intentionally rejects free
    symbols and unsupported natural-language constructions rather than guessing.
    """

    raw = normalize_text(value)
    if not raw:
        return None

    try:
        import sympy as sp  # type: ignore

        text = raw
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = text.replace(r"\,", "").replace(",", "")
        text = text.replace(r"\cdot", "*").replace(r"\times", "*")
        text = text.replace("−", "-")
        text = text.strip().strip("$")
        for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()

        # Accept compact assignments such as ``p = -1`` by taking the RHS.
        if "=" in text:
            text = text.rsplit("=", 1)[1].strip()

        text = _replace_latex_fractions(text)
        text = text.replace("{", "(").replace("}", ")")
        text = text.replace("^", "**")
        if re.search(r"[^0-9+\-*/().\s]", text):
            return None

        parsed = sp.sympify(text, rational=True)
        if getattr(parsed, "free_symbols", set()):
            return None
        simplified = sp.cancel(parsed)
        if simplified.is_Rational:
            return simplified
        return None
    except Exception:
        return None


def exact_numeric_equivalent(left: Any, right: Any) -> bool:
    """Return exact equality without decimal tolerances."""

    parsed_left = parse_exact_numeric_expression(left)
    parsed_right = parse_exact_numeric_expression(right)
    return (
        parsed_left is not None
        and parsed_right is not None
        and parsed_left == parsed_right
    )


def validate_teacher_candidate(
    record: dict[str, Any],
    *,
    converter: BaseIntegerConverter,
) -> tuple[bool, str]:
    """Apply construction-time quality rules to one expert generation."""

    visible = normalize_text(record.get("visible_response") or record.get("response"))
    if not visible:
        return False, "empty_visible_response"
    if bool(record.get("hit_token_limit")):
        return False, "hit_token_limit"
    if not bool(record.get("correct")):
        return False, "failed_openr1_verify"

    boxes, malformed = extract_all_boxed(visible)
    if malformed:
        return False, "malformed_boxed_answer"
    if len(boxes) != 1:
        return False, f"boxed_answer_count:{len(boxes)}"
    if not exact_numeric_equivalent(record.get("answer"), boxes[0]):
        return False, "boxed_answer_not_exactly_equal_to_gold"

    question = normalize_text(record.get("question"))
    question_has_decimals = bool(converter.decimal_tokens(question))
    response_decimals = converter.decimal_tokens(visible)

    try:
        if question_has_decimals:
            # Decimal-containing source tasks use the same conservative limit in
            # the question, compact answer, and generated solution.
            converter.validate_decimal_limit(visible, label="teacher_response")
        else:
            # Any non-integer decimal introduced by the teacher is allowed only
            # when it has an exact finite radix representation in the target base.
            # This keeps harmless values such as 5.000000 but rejects rounded
            # approximations such as 0.00196386 in a base-9 dataset.
            for token in response_decimals:
                if not converter.decimal_has_finite_target_representation(token):
                    return False, f"teacher_introduced_nonfinite_decimal:{token}"

        # Finally ensure the full response is transformable using solution-style
        # LaTeX fractions. This catches unsupported scientific notation early.
        converter.direct_text(
            visible,
            label="teacher_response",
            fraction_style="latex",
            parenthesize_plain_fractions=False,
            enforce_decimal_limit=question_has_decimals,
        )
    except UnsupportedNumericNotationError as exc:
        return False, f"teacher_response_not_transformable:{exc}"

    return True, "accepted"


def filter_source_examples_for_numeric_policy(
    examples: Sequence[SourceExample],
    *,
    converter: BaseIntegerConverter,
) -> tuple[list[SourceExample], dict[str, Any]]:
    """Filter source rows before expensive teacher generation."""

    kept: list[SourceExample] = []
    rejection_counts: Counter[str] = Counter()
    for example in examples:
        try:
            converter.validate_decimal_limit(
                example.question, label=f"source_question:{example.source_id}"
            )
            converter.validate_decimal_limit(
                example.answer, label=f"source_answer:{example.source_id}"
            )
            # Validate transformability and desired output styles without storing
            # the converted values yet.
            converter.direct_text(
                example.question,
                label=f"source_question:{example.source_id}",
                fraction_style="plain",
                parenthesize_plain_fractions=True,
            )
            converter.direct_text(
                example.answer,
                label=f"source_answer:{example.source_id}",
                fraction_style="plain",
                parenthesize_plain_fractions=False,
            )
        except UnsupportedNumericNotationError as exc:
            rejection_counts[str(exc)] += 1
            continue
        kept.append(example)

    return kept, {
        "num_before": len(examples),
        "num_after": len(kept),
        "num_rejected": len(examples) - len(kept),
        "rejection_counts": dict(rejection_counts.most_common()),
    }


def visible_response_for_scoring(text: Any) -> tuple[str, str | None]:
    """Remove closed thinking blocks and reject an unclosed thinking block."""

    raw = str(text or "")
    has_open = bool(re.search(r"<think\b[^>]*>", raw, flags=re.IGNORECASE))
    has_close = bool(re.search(r"</think\s*>", raw, flags=re.IGNORECASE))
    if has_open and not has_close:
        return "", "unclosed think block"
    return _THINK_BLOCK_RE.sub("", raw).strip(), None


class BaseAwareMathVerifier:
    """Delegate original-world scoring and back-convert mod-world answers.

    For the transformed world, only the final boxed expression and the compact
    reference are converted back to decimal notation. The surrounding reasoning
    is intentionally left untouched, avoiding accidental conversion of section
    numbers or prose. Any invalid base digit or unsupported notation yields a
    deterministic incorrect result rather than falling back to decimal scoring.
    """

    def __init__(self, scorer: Any, converter: BaseIntegerConverter) -> None:
        self.scorer = scorer
        self.converter = converter

    def verify_with_details(
        self,
        gold: Any,
        prediction: Any,
        *,
        world: str,
    ) -> Any:
        if world == "original":
            return self.scorer.verify_with_details(
                gold,
                prediction,
                strip_thinking=True,
                require_visible_after_think=False,
            )
        if world != "mod":
            raise ValueError(f"Unknown verification world: {world!r}")

        visible, structural_error = visible_response_for_scoring(prediction)
        if structural_error:
            return BaseAwareVerificationResult(
                correct=False,
                method="base_backconversion_failed",
                error=structural_error,
            )

        boxed = extract_last_boxed(visible)
        if boxed is None:
            return BaseAwareVerificationResult(
                correct=False,
                method="base_backconversion_failed",
                error="no boxed answer found",
            )

        try:
            decimal_gold = self.converter.reverse_text(gold, label="mod_gold")
            decimal_prediction = self.converter.reverse_text(
                boxed, label="mod_boxed_prediction"
            )
        except UnsupportedNumericNotationError as exc:
            return BaseAwareVerificationResult(
                correct=False,
                method="base_backconversion_failed",
                extracted_prediction=boxed,
                error=str(exc),
            )

        delegated = self.scorer.verify_with_details(
            decimal_gold,
            rf"\boxed{{{decimal_prediction}}}",
            strip_thinking=True,
            require_visible_after_think=False,
        )
        parsed_gold = parse_exact_numeric_expression(decimal_gold)
        parsed_prediction = parse_exact_numeric_expression(decimal_prediction)
        if parsed_gold is not None and parsed_prediction is not None:
            correct = parsed_gold == parsed_prediction
            method = f"base{self.converter.base}_backconvert:strict_exact_numeric"
        else:
            correct = bool(delegated.correct)
            method = f"base{self.converter.base}_backconvert:{delegated.method}"
        return BaseAwareVerificationResult(
            correct=correct,
            method=method,
            extracted_prediction=boxed,
            gold_parsed=getattr(delegated, "gold_parsed", ""),
            prediction_parsed=getattr(delegated, "prediction_parsed", ""),
            error=getattr(delegated, "error", ""),
            decimal_gold=decimal_gold,
            decimal_prediction=decimal_prediction,
        )


@dataclass(frozen=True)
class SourceExample:
    """One source problem before expert demonstration generation."""

    row_idx: int
    source_id: str
    question: str
    answer: str
    module: str
    difficulty: str
    source: str
    bucket: str


@dataclass(frozen=True)
class VLLMCompletion:
    """One model completion returned by vLLM."""

    row_idx: int
    response_idx: int
    text: str
    generated_num_tokens: int
    finish_reason: str
    hit_token_limit: bool
    finish_eos: bool


def build_openr1_messages(problem: str) -> list[dict[str, str]]:
    """Build the canonical OpenR1 user-only prompt shell."""

    return [
        {
            "role": "user",
            "content": OPENR1_USER_PROMPT_TEMPLATE.format(problem=normalize_text(problem)),
        }
    ]


def apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    """Render a chat prompt, supporting tokenizers with/without enable_thinking."""

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_vllm_model(
    *,
    model_name: str,
    tokenizer_name: str | None,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    tensor_parallel_size: int,
    dtype: str,
    seed: int,
) -> tuple[Any, Any]:
    """Load a vLLM model and tokenizer lazily."""

    from transformers import AutoTokenizer  # type: ignore
    from vllm import LLM  # type: ignore

    effective_tokenizer = tokenizer_name or model_name
    logging.info("Loading tokenizer: %s", effective_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        effective_tokenizer,
        padding_side="left",
        trust_remote_code=True,
    )

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": effective_tokenizer,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "dtype": dtype,
        "trust_remote_code": True,
        "tensor_parallel_size": int(tensor_parallel_size),
        "seed": int(seed),
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)

    logging.info(
        "Loading vLLM model model=%s gpu_memory_utilization=%.3f max_model_len=%s tensor_parallel_size=%d dtype=%s",
        model_name,
        gpu_memory_utilization,
        max_model_len,
        tensor_parallel_size,
        dtype,
    )
    llm = LLM(**llm_kwargs)
    return llm, tokenizer


def cleanup_cuda() -> None:
    """Best-effort Python/CUDA cleanup."""

    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def destroy_vllm_engine(llm: Any) -> None:
    """Best-effort vLLM cleanup before loading another model.

    vLLM shutdown APIs have changed across versions. This helper tries several
    known paths, destroys distributed/model-parallel state when available, and
    finally clears Python/CUDA caches. It is intentionally tolerant to missing
    methods.
    """

    if llm is not None:
        for attr_chain in (("shutdown",), ("llm_engine", "shutdown"), ("engine", "shutdown")):
            try:
                obj = llm
                for attr in attr_chain:
                    obj = getattr(obj, attr)
                if callable(obj):
                    obj()
            except Exception:
                pass

        try:
            del llm
        except Exception:
            pass

    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    cleanup_cuda()


def generate_with_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    prompts: list[str],
    row_indices: list[int],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    truncate_prompt_tokens: int | None,
) -> list[VLLMCompletion]:
    """Generate completions for a batch of prompts using vLLM."""

    from vllm import SamplingParams  # type: ignore

    sampling_kwargs: dict[str, Any] = {
        "n": int(n),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
        "seed": int(seed),
    }
    if temperature <= 0:
        sampling_kwargs["temperature"] = 0.0
        sampling_kwargs["top_p"] = 1.0
    if getattr(tokenizer, "eos_token_id", None) is not None:
        sampling_kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    if truncate_prompt_tokens is not None and truncate_prompt_tokens > 0:
        sampling_kwargs["truncate_prompt_tokens"] = int(truncate_prompt_tokens)

    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(**sampling_kwargs),
        use_tqdm=False,
    )

    completions: list[VLLMCompletion] = []
    for local_i, out in enumerate(outputs):
        row_idx = int(row_indices[local_i])
        for response_idx, completion in enumerate(out.outputs):
            token_ids = [int(x) for x in getattr(completion, "token_ids", [])]
            finish_reason = str(getattr(completion, "finish_reason", "") or "")
            finish_eos = bool(
                token_ids
                and getattr(tokenizer, "eos_token_id", None) is not None
                and token_ids[-1] == tokenizer.eos_token_id
            ) or finish_reason == "stop"
            hit_limit = bool(
                finish_reason == "length"
                or (len(token_ids) >= int(max_new_tokens) and not finish_eos)
            )
            completions.append(
                VLLMCompletion(
                    row_idx=row_idx,
                    response_idx=int(response_idx),
                    text=str(getattr(completion, "text", "") or ""),
                    generated_num_tokens=len(token_ids),
                    finish_reason=finish_reason,
                    hit_token_limit=hit_limit,
                    finish_eos=finish_eos,
                )
            )
    return completions


def standardize_source_row(row: dict[str, Any], row_idx: int) -> SourceExample:
    """Map several likely source schemas to SourceExample."""

    question = normalize_text(
        row.get("question")
        or row.get("problem")
        or row.get("prompt")
        or row.get("input")
        or row.get("messages", "")
    )
    answer = normalize_text(row.get("answer") or row.get("golden_answer") or row.get("target"))
    if not question or not answer:
        raise ValueError(f"Source row {row_idx} is missing question/problem or answer: {row}")

    source_id = normalize_text(row.get("id") or row.get("source_id") or row.get("id_in_dataset"))
    if not source_id:
        source_id = f"source_{stable_hash(row_idx, question, answer)}"

    module = normalize_text(row.get("module") or row.get("problem_type") or row.get("source_module") or "unknown_module")
    difficulty = normalize_text(row.get("difficulty") or row.get("question_type") or row.get("source_difficulty") or "unknown_difficulty")
    source = normalize_text(row.get("source") or row.get("dataset_name") or "deepmind_mathematics_dataset_v1.0")
    bucket = f"{module}/{difficulty}"
    return SourceExample(
        row_idx=int(row_idx),
        source_id=source_id,
        question=question,
        answer=answer,
        module=module,
        difficulty=difficulty,
        source=source,
        bucket=bucket,
    )


def load_source_examples(input_path: Path, *, limit: int | None = None) -> list[SourceExample]:
    """Load source examples from a JSONL file, directory, or HF dataset path."""

    path = input_path.expanduser()
    raw_rows: list[dict[str, Any]]

    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            raw_rows = read_jsonl(path)
        elif path.suffix.lower() == ".json":
            data = read_json(path)
            if isinstance(data, list):
                raw_rows = [dict(x) for x in data]
            else:
                raise ValueError(f"Expected a JSON list in {path}")
        else:
            raise ValueError(f"Unsupported input file extension: {path}")
    elif path.is_dir():
        jsonl_path = path / "filtered_math_dataset.jsonl"
        if jsonl_path.exists():
            raw_rows = read_jsonl(jsonl_path)
        else:
            try:
                from datasets import load_from_disk  # type: ignore

                dataset = load_from_disk(str(path))
                raw_rows = dataset.to_list()
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not find filtered_math_dataset.jsonl or a load_from_disk-compatible dataset at {path}"
                ) from exc
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if limit is not None:
        raw_rows = raw_rows[: int(limit)]

    examples: list[SourceExample] = []
    for i, row in enumerate(raw_rows):
        examples.append(standardize_source_row(row, i))

    logging.info("Loaded %d source examples from %s", len(examples), path)
    return examples


def generation_record_path(output_root: Path) -> Path:
    """Return the expert generation JSONL path."""

    return output_root / "expert_generation_records.jsonl"


def accepted_record_path(output_root: Path) -> Path:
    """Return the transformed accepted JSONL path."""

    return output_root / "accepted_transformed_records.jsonl"


def load_existing_generated_source_ids(
    path: Path,
    *,
    required_generations: int,
) -> set[str]:
    """Return source IDs with enough complete generation attempts for resume.

    A completed attempt need not pass the mathematical or transformation policy:
    two completed attempts are enough to make the keep/discard decision stable.
    Truncated or empty attempts do not count and are regenerated.
    """

    counts: Counter[str] = Counter()
    for record in read_jsonl(path):
        source_id = normalize_text(record.get("source_id"))
        visible = normalize_text(record.get("visible_response") or record.get("response"))
        if source_id and visible and not bool(record.get("hit_token_limit")):
            counts[source_id] += 1
    return {
        source_id
        for source_id, count in counts.items()
        if count >= int(required_generations)
    }


def render_source_prompts(
    examples: Sequence[SourceExample],
    tokenizer: Any,
    *,
    enable_thinking: bool | None,
) -> tuple[list[str], list[int]]:
    """Render OpenR1 prompts for source examples."""

    prompts: list[str] = []
    row_indices: list[int] = []
    for ex in examples:
        prompts.append(
            apply_chat_template_compat(
                tokenizer,
                build_openr1_messages(ex.question),
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        row_indices.append(ex.row_idx)
    return prompts, row_indices


def generate_expert_demonstrations(
    *,
    source_examples: list[SourceExample],
    scorer: Any,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate, verify, and append expert demonstration records."""

    records_path = generation_record_path(output_root)
    existing_ids = (
        load_existing_generated_source_ids(
            records_path, required_generations=args.expert_num_generations
        )
        if args.resume
        else set()
    )
    if existing_ids:
        logging.info("Resume enabled: found %d source IDs with enough complete expert attempts", len(existing_ids))

    examples_to_run = [ex for ex in source_examples if ex.source_id not in existing_ids]
    logging.info(
        "Expert generation: %d remaining / %d total source examples",
        len(examples_to_run),
        len(source_examples),
    )
    if not examples_to_run:
        return {
            "source_rows": len(source_examples),
            "remaining_generated": 0,
            "existing_source_ids": len(existing_ids),
            "records_path": str(records_path),
        }

    llm = None
    tokenizer = None
    start_time = time.time()
    try:
        llm, tokenizer = load_vllm_model(
            model_name=args.expert_model,
            tokenizer_name=args.expert_tokenizer,
            gpu_memory_utilization=args.expert_gpu_memory_utilization,
            max_model_len=args.expert_max_model_len,
            tensor_parallel_size=args.expert_tensor_parallel_size,
            dtype=args.expert_dtype,
            seed=args.seed,
        )

        by_idx = {ex.row_idx: ex for ex in source_examples}
        num_written = 0
        for batch_no, batch in enumerate(chunks(examples_to_run, args.generation_batch_size), start=1):
            prompts, row_indices = render_source_prompts(
                list(batch),
                tokenizer,
                enable_thinking=False if args.expert_disable_thinking else None,
            )
            completions = generate_with_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                row_indices=row_indices,
                n=args.expert_num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.expert_temperature,
                top_p=args.expert_top_p,
                seed=args.seed + batch_no,
                truncate_prompt_tokens=args.max_prompt_tokens,
            )

            out_records: list[dict[str, Any]] = []
            for completion in completions:
                ex = by_idx[completion.row_idx]
                visible = args.extract_visible_response(completion.text)
                details = scorer.verify_with_details(
                    ex.answer,
                    completion.text,
                    strip_thinking=True,
                    require_visible_after_think=True,
                )
                out_records.append(
                    {
                        "created_at": now(),
                        "row_idx": ex.row_idx,
                        "source_id": ex.source_id,
                        "candidate_idx": completion.response_idx,
                        "question": ex.question,
                        "answer": ex.answer,
                        "module": ex.module,
                        "difficulty": ex.difficulty,
                        "bucket": ex.bucket,
                        "source": ex.source,
                        "response": completion.text,
                        "visible_response": visible.text,
                        "had_think_block": visible.had_think_block,
                        "has_unclosed_think": visible.has_unclosed_think,
                        "generated_num_tokens": completion.generated_num_tokens,
                        "finish_reason": completion.finish_reason,
                        "hit_token_limit": completion.hit_token_limit,
                        "finish_eos": completion.finish_eos,
                        "correct": bool(details.correct),
                        "verify_method": details.method,
                        "verify_extracted_prediction": details.extracted_prediction,
                        "verify_error": details.error,
                    "verify_decimal_gold": normalize_text(getattr(details, "decimal_gold", "")),
                    "verify_decimal_prediction": normalize_text(getattr(details, "decimal_prediction", "")),
                    }
                )

            num_written += append_jsonl(records_path, out_records)
            if batch_no == 1 or batch_no % args.log_every_batches == 0:
                recent_correct = sum(int(r["correct"]) for r in out_records)
                logging.info(
                    "Expert batch %d: wrote %d records, recent candidate_acc=%.3f, total_written=%d",
                    batch_no,
                    len(out_records),
                    recent_correct / max(1, len(out_records)),
                    num_written,
                )

        return {
            "source_rows": len(source_examples),
            "remaining_generated": len(examples_to_run),
            "existing_source_ids": len(existing_ids),
            "records_written_this_run": num_written,
            "records_path": str(records_path),
            "elapsed_seconds": time.time() - start_time,
        }
    finally:
        logging.info("Destroying expert model before continuing.")
        destroy_vllm_engine(llm)
        try:
            del tokenizer
        except Exception:
            pass
        cleanup_cuda()


def choose_best_correct_generations(
    generation_records: list[dict[str, Any]],
    *,
    converter: BaseIntegerConverter,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select the shortest candidate that satisfies every construction rule."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in generation_records:
        source_id = normalize_text(record.get("source_id"))
        if source_id:
            grouped[source_id].append(record)

    selected: dict[str, dict[str, Any]] = {}
    rejection_counts: Counter[str] = Counter()
    sources_with_no_valid_candidate = 0

    for source_id, candidates in grouped.items():
        valid: list[dict[str, Any]] = []
        for candidate in candidates:
            accepted, reason = validate_teacher_candidate(
                candidate,
                converter=converter,
            )
            candidate["construction_policy_valid"] = bool(accepted)
            candidate["construction_policy_reason"] = reason
            if accepted:
                valid.append(candidate)
            else:
                rejection_counts[reason] += 1

        if not valid:
            sources_with_no_valid_candidate += 1
            continue

        valid.sort(
            key=lambda candidate: (
                int(candidate.get("generated_num_tokens", 10**9)),
                int(candidate.get("candidate_idx", 10**9)),
                str(candidate.get("created_at", "")),
            )
        )
        selected[source_id] = valid[0]

    return selected, {
        "num_source_groups": len(grouped),
        "num_selected": len(selected),
        "num_sources_without_valid_candidate": sources_with_no_valid_candidate,
        "candidate_rejection_counts": dict(rejection_counts.most_common()),
    }


def _strip_outer_problem_text(problem: str) -> str:
    """Extract the mathematical expression part from simple arithmetic prompts.

    This is intentionally conservative: it only handles templates that look like
    direct arithmetic calculation requests. If extraction is uncertain, the
    caller should return None instead of rejecting the example.
    """

    text = normalize_text(problem)
    text = text.strip().rstrip("?.")

    # Common DeepMind arithmetic templates.
    patterns = [
        r"^Calculate\s+(.+)$",
        r"^What\s+is\s+(.+)$",
        r"^Evaluate\s+(.+)$",
        r"^Find\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text


def _normalize_arithmetic_expression(expr: str) -> str:
    """Normalize a simple English/LaTeX-ish arithmetic expression for SymPy."""

    out = normalize_text(expr)
    out = out.replace("−", "-").replace("–", "-").replace("—", "-")
    out = out.replace("×", "*").replace("·", "*").replace("÷", "/")
    out = re.sub(r"\bdivided\s+by\b", "/", out, flags=re.IGNORECASE)
    out = re.sub(r"\btimes\b", "*", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmultiplied\s+by\b", "*", out, flags=re.IGNORECASE)
    out = re.sub(r"\bplus\b", "+", out, flags=re.IGNORECASE)
    out = re.sub(r"\bminus\b", "-", out, flags=re.IGNORECASE)
    out = out.replace("^", "**")
    out = out.replace("$", "")
    out = out.strip()
    return out


def _sympy_number_to_answer(value: Any) -> str | None:
    """Convert a SymPy numeric result into a compact answer string."""

    try:
        import sympy as sp  # type: ignore
    except Exception:
        return None

    try:
        value = sp.simplify(value)
        if value.is_Integer:
            return str(int(value))
        if value.is_Rational:
            return f"{int(value.p)}/{int(value.q)}"
        if value.is_number:
            return str(value)
    except Exception:
        return None
    return None


def solve_arithmetic_problem_under_decimal(problem: str) -> tuple[str | None, str | None]:
    """Try to solve a transformed problem using ordinary decimal arithmetic.

    Returns (answer, method). If the problem is not a simple arithmetic expression
    recognized by this conservative parser, returns (None, None).
    """

    try:
        import sympy as sp  # type: ignore
    except Exception:
        return None, None

    expr = _strip_outer_problem_text(problem)
    expr = _normalize_arithmetic_expression(expr)

    # Reject anything that is not a plain arithmetic expression. This prevents
    # false rejections on natural-language/algebra prompts.
    if not re.fullmatch(r"[0-9\s+\-*/().]+", expr):
        return None, None
    if "=" in expr:
        return None, None

    try:
        value = sp.sympify(expr, evaluate=True)
        answer = _sympy_number_to_answer(value)
    except Exception:
        return None, None
    if answer is None:
        return None, None
    return answer, "sympy_arithmetic_decimal"


def solve_linear_problem_under_decimal(problem: str) -> tuple[str | None, str | None]:
    """Solve conservative one- or two-equation linear prompts as decimal math."""

    try:
        import sympy as sp  # type: ignore
    except Exception:
        return None, None

    text = normalize_text(problem).strip().rstrip("?.")
    match = re.match(r"^Solve\s+(.+?)\s+for\s+([A-Za-z])$", text, flags=re.IGNORECASE)
    if not match:
        return None, None

    equations_text = _normalize_arithmetic_expression(match.group(1))
    target_name = match.group(2)
    equation_parts = [part.strip() for part in equations_text.split(",")]
    if not equation_parts or len(equation_parts) > 2:
        return None, None
    if any(part.count("=") != 1 for part in equation_parts):
        return None, None

    variable_names = sorted(set(re.findall(r"\b[A-Za-z]\b", equations_text)))
    if target_name not in variable_names or len(variable_names) > 2:
        return None, None

    allowed = r"[0-9A-Za-z\s+\-*/().]+"
    symbols = {name: sp.Symbol(name) for name in variable_names}
    equations: list[Any] = []
    try:
        for equation_text in equation_parts:
            lhs_text, rhs_text = [part.strip() for part in equation_text.split("=", 1)]
            if not re.fullmatch(allowed, lhs_text) or not re.fullmatch(allowed, rhs_text):
                return None, None
            lhs = sp.sympify(lhs_text, locals=symbols, evaluate=True)
            rhs = sp.sympify(rhs_text, locals=symbols, evaluate=True)
            equations.append(sp.Eq(lhs, rhs))
        solutions = sp.solve(equations, list(symbols.values()), dict=True)
    except Exception:
        return None, None

    if len(solutions) != 1:
        return None, None
    target = symbols[target_name]
    if target not in solutions[0]:
        return None, None
    answer = _sympy_number_to_answer(solutions[0][target])
    if answer is None:
        return None, None
    method = "sympy_linear_2d_decimal" if len(equation_parts) == 2 else "sympy_linear_decimal"
    return answer, method


def solve_problem_under_decimal(problem: str, module: str) -> tuple[str | None, str | None]:
    """Try to solve a transformed problem as if its numerals were decimal.

    This is used only as a filter. If the ordinary-decimal answer equals the
    transformed/base-N target, the example is not a contradiction and should be
    removed. Unrecognized templates are not rejected.
    """

    module = normalize_text(module)
    if module.startswith("arithmetic__"):
        return solve_arithmetic_problem_under_decimal(problem)
    if module.startswith("algebra__linear_"):
        return solve_linear_problem_under_decimal(problem)
    return None, None


def detect_degenerate_identity_pattern(problem: str) -> str | None:
    """Detect obvious identity/annihilator cases that often survive conversion.

    This is intentionally conservative and complements the symbolic decimal
    check. It catches examples like "N divided by 1" even if parsing fails.
    """

    text = normalize_text(problem).lower()
    compact = re.sub(r"\s+", "", text)
    if re.search(r"\bdivided\s+by\s+1\b", text) or re.search(r"/\s*1(?!\d)", text):
        return "division_by_one"
    if re.search(r"\btimes\s+1\b", text) or re.search(r"\bmultiplied\s+by\s+1\b", text) or re.search(r"\*\s*1(?!\d)", text):
        return "multiplication_by_one"
    if re.search(r"\btimes\s+0\b", text) or re.search(r"\bmultiplied\s+by\s+0\b", text) or re.search(r"\*\s*0(?!\d)", text):
        return "multiplication_by_zero"
    if re.search(r"(?<!\d)0\s*\*", text):
        return "zero_times_expression"
    if "+0" in compact or "+-0" in compact:
        return "addition_of_zero"
    if "-0" in compact:
        return "subtraction_of_zero"
    return None


def transformed_answer_is_decimal_consistent(
    *,
    mod_problem: str,
    mod_answer: str,
    module: str,
    scorer: Any,
) -> tuple[bool, str, str | None]:
    """Return whether ordinary decimal solving also gives the transformed answer."""

    decimal_answer, method = solve_problem_under_decimal(mod_problem, module)
    if decimal_answer is None:
        return False, "unresolved", None

    details = scorer.verify_with_details(
        mod_answer,
        f"\\boxed{{{decimal_answer}}}",
        strip_thinking=True,
        require_visible_after_think=False,
    )
    if details.correct:
        return True, method or "decimal_solver", decimal_answer
    return False, method or "decimal_solver", decimal_answer

def transform_one_record(
    *,
    selected: dict[str, Any],
    converter: BaseIntegerConverter,
    scorer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    """Transform one verified expert generation into the final raw schema."""

    original_problem = normalize_text(selected.get("question"))
    original_answer = normalize_text(selected.get("answer"))
    original_solution = normalize_text(selected.get("visible_response") or selected.get("response"))
    source_id = normalize_text(selected.get("source_id"))

    if not original_problem or not original_answer or not original_solution:
        return None, "missing_original_fields"

    try:
        mod_problem = normalize_text(
            converter.direct_text(
                original_problem,
                label=f"problem:{source_id}",
                fraction_style="plain",
                parenthesize_plain_fractions=True,
                enforce_decimal_limit=True,
            )
        )
        mod_answer = normalize_text(
            converter.direct_text(
                original_answer,
                label=f"answer:{source_id}",
                fraction_style="plain",
                parenthesize_plain_fractions=False,
                enforce_decimal_limit=True,
            )
        )
        question_has_decimals = bool(converter.decimal_tokens(original_problem))
        mod_solution = normalize_text(
            converter.direct_text(
                original_solution,
                label=f"solution:{source_id}",
                fraction_style="latex",
                parenthesize_plain_fractions=False,
                enforce_decimal_limit=question_has_decimals,
            )
        )
    except UnsupportedNumericNotationError as exc:
        return None, f"unsupported_numeric_notation:{exc}"

    if mod_problem == original_problem:
        return None, "unchanged_problem"

    mod_equals_original_answer = mod_answer == original_answer
    if mod_equals_original_answer and bool(
        getattr(args, "drop_unchanged_final_answer", False)
    ):
        return None, "unchanged_final_answer"

    # Verify transformed answers under base-N semantics by converting only the
    # boxed expression and compact reference back to decimal at runtime.
    mod_verifier = BaseAwareMathVerifier(scorer, converter)
    mod_details = mod_verifier.verify_with_details(
        mod_answer,
        mod_solution,
        world="mod",
    )
    if not mod_details.correct:
        return None, f"mod_solution_failed_verify:{mod_details.method}:{mod_details.error}"
    if not exact_numeric_equivalent(
        getattr(mod_details, "decimal_gold", ""),
        getattr(mod_details, "decimal_prediction", ""),
    ):
        return None, "mod_solution_failed_strict_exact_verify"

    module = normalize_text(selected.get("module"))

    # Remove examples that remain correct under ordinary decimal arithmetic.
    # These are not useful contradictions because an unadapted model can solve
    # them without discovering the hidden base-N notation.
    if bool(getattr(args, "drop_degenerate_identity_patterns", True)):
        degenerate_reason = detect_degenerate_identity_pattern(mod_problem)
        if degenerate_reason:
            return None, f"degenerate_identity:{degenerate_reason}"

    decimal_consistent = False
    decimal_solver_method = ""
    decimal_interpretation_answer = None
    if bool(getattr(args, "drop_decimal_consistent_mod", True)):
        decimal_consistent, decimal_solver_method, decimal_interpretation_answer = transformed_answer_is_decimal_consistent(
            mod_problem=mod_problem,
            mod_answer=mod_answer,
            module=module,
            scorer=scorer,
        )
        if decimal_consistent:
            return None, f"decimal_consistent_mod:{decimal_solver_method}"

    problem_hash = stable_hash("math_contradiction", source_id, mod_problem, mod_answer)
    original_messages = build_openr1_messages(original_problem)
    mod_messages = build_openr1_messages(mod_problem)

    record = {
        # OpenR1-like default columns: transformed/base-N world.
        "messages": mod_messages,
        "problem": mod_problem,
        "answer": mod_answer,
        "output_text": mod_solution,
        "visible_output_text": mod_solution,
        "golden_answer": mod_answer,
        "golden_response": mod_solution,
        "question": mod_problem,
        # Explicit transformed aliases.
        "mod_problem": mod_problem,
        "mod_answer": mod_answer,
        "mod_output_text": mod_solution,
        "mod_visible_output_text": mod_solution,
        "mod_messages": mod_messages,
        # Original/decimal world preserved for controls and student eval.
        "original_problem": original_problem,
        "original_answer": original_answer,
        "original_output_text": original_solution,
        "original_visible_output_text": original_solution,
        "original_golden_answer": original_answer,
        "original_golden_response": original_solution,
        "original_messages": original_messages,
        # Metadata/provenance.
        "dataset_name": "math_contradiction",
        "source_dataset_name": "deepmind_mathematics_dataset_v1.0_filtered",
        "source": normalize_text(selected.get("source")),
        "source_id": source_id,
        "id_in_dataset": source_id,
        "uuid": source_id,
        "problem_hash": problem_hash,
        "module": module,
        "difficulty": normalize_text(selected.get("difficulty")),
        "bucket": normalize_text(selected.get("bucket")),
        "problem_type": normalize_text(selected.get("module")),
        "question_type": normalize_text(selected.get("difficulty")),
        "base": converter.base,
        "transformation": "numeric_literals_decimal_to_base_notation",
        "transformation_version": TRANSFORMATION_VERSION,
        "mod_equals_original_answer": mod_equals_original_answer,
        "decimal_consistency_filter_enabled": bool(getattr(args, "drop_decimal_consistent_mod", True)),
        "decimal_interpretation_answer": normalize_text(decimal_interpretation_answer),
        "decimal_interpretation_solver": normalize_text(decimal_solver_method),
        # Verification/generation provenance.
        "correctness_math_verify": True,
        "correctness_llama": False,
        "is_reasoning_complete": True,
        "generation_idx": int(selected.get("candidate_idx", 0)),
        "finish_reason": normalize_text(selected.get("finish_reason")),
        "correctness_count": -1,
        "prompt_tokens": -1,
        "output_tokens": int(selected.get("generated_num_tokens", -1)),
        "local_verify_method": normalize_text(selected.get("verify_method")),
        "local_extracted_answer": normalize_text(selected.get("verify_extracted_prediction")),
        "mod_local_verify_method": mod_details.method,
        "mod_local_extracted_answer": mod_details.extracted_prediction,
        "mod_decimal_gold": normalize_text(getattr(mod_details, "decimal_gold", "")),
        "mod_decimal_prediction": normalize_text(getattr(mod_details, "decimal_prediction", "")),
        "teacher_candidate_policy_valid": bool(selected.get("construction_policy_valid", True)),
        "teacher_candidate_policy_reason": normalize_text(selected.get("construction_policy_reason", "accepted")),
        "teacher_question_had_decimals": bool(converter.decimal_tokens(original_problem)),
    }
    return record, "accepted"


def build_transformed_records(
    *,
    output_root: Path,
    scorer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Read expert generations, transform verified examples, and write accepted JSONL."""

    out_path = accepted_record_path(output_root)
    if out_path.exists() and args.resume and not args.force_rebuild_transforms:
        existing = read_jsonl(out_path)
        current_version = bool(existing) and all(
            int(record.get("transformation_version", -1)) == TRANSFORMATION_VERSION
            for record in existing
        )
        if current_version:
            logging.info(
                "Using existing transformed records because --resume is set: %s (%d rows)",
                out_path,
                len(existing),
            )
            return {
                "accepted_records_path": str(out_path),
                "accepted": len(existing),
                "used_existing": True,
                "transformation_version": TRANSFORMATION_VERSION,
            }
        logging.warning(
            "Existing transformed records use an older conversion/scoring schema; rebuilding %s",
            out_path,
        )

    if out_path.exists():
        out_path.unlink()

    generation_records = read_jsonl(generation_record_path(output_root))
    converter = BaseIntegerConverter(
        base=args.base,
        max_decimal_places=args.max_decimal_places,
        max_finite_fractional_digits=args.max_finite_fractional_digits,
    )
    selected, selection_summary = choose_best_correct_generations(
        generation_records,
        converter=converter,
    )
    logging.info(
        "Selected %d source examples with at least one policy-valid expert generation from %d candidate records",
        len(selected),
        len(generation_records),
    )
    if selection_summary.get("candidate_rejection_counts"):
        logging.info(
            "Teacher candidate rejection counts: %s",
            selection_summary["candidate_rejection_counts"],
        )
    rejection_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    for source_id, selected_record in selected.items():
        record, reason = transform_one_record(
            selected=selected_record,
            converter=converter,
            scorer=scorer,
            args=args,
        )
        if record is None:
            rejection_counts[reason] += 1
            continue
        accepted.append(record)

    # Stable shuffle after filtering.
    accepted.sort(key=lambda r: stable_float("accepted_order", args.seed, r["problem_hash"]))
    if args.max_final_rows is not None:
        accepted = accepted[: int(args.max_final_rows)]

    append_jsonl(out_path, accepted)
    logging.info("Wrote %d transformed accepted records to %s", len(accepted), out_path)
    if rejection_counts:
        logging.info("Transform rejection counts: %s", dict(rejection_counts.most_common()))

    return {
        "accepted_records_path": str(out_path),
        "candidate_generation_records": len(generation_records),
        "selected_correct_source_examples": len(selected),
        "teacher_candidate_selection": selection_summary,
        "accepted": len(accepted),
        "rejection_counts": dict(rejection_counts),
        "used_existing": False,
    }


def stratified_train_eval_split(
    records: list[dict[str, Any]], *, eval_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create an approximately stratified split by module/difficulty bucket."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    if not records:
        raise ValueError("No records available for splitting")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("bucket") or "unknown")].append(record)

    train: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    bucket_summary: dict[str, Any] = {}

    for bucket, bucket_records in sorted(groups.items()):
        ordered = sorted(
            bucket_records,
            key=lambda r: stable_float("split", seed, bucket, r.get("problem_hash", r.get("source_id", ""))),
        )
        if len(ordered) <= 1:
            n_eval = 0
        else:
            n_eval = max(1, int(round(len(ordered) * eval_fraction)))
            n_eval = min(n_eval, len(ordered) - 1)
        bucket_eval = ordered[:n_eval]
        bucket_train = ordered[n_eval:]
        eval_records.extend(bucket_eval)
        train.extend(bucket_train)
        bucket_summary[bucket] = {
            "total": len(ordered),
            "train": len(bucket_train),
            "eval": len(bucket_eval),
        }

    train.sort(key=lambda r: stable_float("train_order", seed, r.get("problem_hash", "")))
    eval_records.sort(key=lambda r: stable_float("eval_order", seed, r.get("problem_hash", "")))

    for split_name, split_records in (("train", train), ("eval", eval_records)):
        for record in split_records:
            record["split"] = split_name

    summary = {
        "eval_fraction_requested": eval_fraction,
        "total": len(records),
        "train": len(train),
        "eval": len(eval_records),
        "actual_eval_fraction": len(eval_records) / max(1, len(records)),
        "bucket_summary": bucket_summary,
    }
    return train, eval_records, summary


def validate_final_records(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    require_changed_final_answer: bool = False,
) -> None:
    """Fail fast on schema/split mistakes.

    ``require_changed_final_answer`` must mirror the construction policy.
    Unchanged compact answers are valid by default because the transformed
    problem can still require base-N reasoning.
    """

    required = {
        "messages",
        "problem",
        "answer",
        "output_text",
        "visible_output_text",
        "golden_answer",
        "golden_response",
        "original_problem",
        "original_answer",
        "original_output_text",
        "mod_problem",
        "mod_answer",
        "problem_hash",
    }
    forbidden = {"prompt", "teacher_prompt", "target", "sft_messages"}
    train_hashes = {r["problem_hash"] for r in train_records}
    eval_hashes = {r["problem_hash"] for r in eval_records}
    leakage = train_hashes & eval_hashes
    if leakage:
        raise RuntimeError(f"Train/eval leakage detected for {len(leakage)} problem hashes")

    for split_name, records in (("train", train_records), ("eval", eval_records)):
        for i, record in enumerate(records[: min(20, len(records))]):
            missing = sorted(required - set(record))
            if missing:
                raise RuntimeError(f"{split_name}[{i}] missing required columns: {missing}")
            present_forbidden = sorted(forbidden & set(record))
            if present_forbidden:
                raise RuntimeError(f"{split_name}[{i}] contains formatted/training-time columns: {present_forbidden}")
            if record["answer"] != record["mod_answer"]:
                raise RuntimeError(f"{split_name}[{i}] answer != mod_answer")
            if record["problem"] != record["mod_problem"]:
                raise RuntimeError(f"{split_name}[{i}] problem != mod_problem")
            if (
                require_changed_final_answer
                and record["answer"] == record["original_answer"]
            ):
                raise RuntimeError(
                    f"{split_name}[{i}] unchanged final answer was not filtered "
                    "despite --drop_unchanged_final_answer"
                )


def save_splits(output_root: Path, train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], *, overwrite_splits: bool) -> None:
    """Save train/eval records as Hugging Face Arrow datasets."""

    from datasets import Dataset  # type: ignore

    for split_name, records in (("train_data", train_records), ("eval_data", eval_records)):
        path = output_root / split_name
        if path.exists():
            if overwrite_splits:
                shutil.rmtree(path)
            else:
                raise FileExistsError(
                    f"Refusing to overwrite existing split {path}. Use --overwrite_splits or --overwrite."
                )
        logging.info("Saving %s to %s (%d rows)", split_name, path, len(records))
        Dataset.from_list(records).save_to_disk(str(path))


def load_built_split(output_root: Path, split: str) -> list[dict[str, Any]]:
    """Load a saved train/eval split from disk."""

    from datasets import load_from_disk  # type: ignore

    path = output_root / ("train_data" if split == "train" else "eval_data")
    if not path.exists():
        raise FileNotFoundError(f"Saved split not found: {path}")
    return load_from_disk(str(path)).to_list()


def count_by_key(records: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    """Count records by one metadata key."""

    return dict(Counter(str(r.get(key, "")) for r in records))


def quantiles(values: list[float]) -> dict[str, float | None]:
    """Return common quantiles for metadata summaries."""

    if not values:
        return {"min": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    xs = sorted(float(v) for v in values)

    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    return {"min": xs[0], "p50": q(0.50), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": xs[-1]}


def safe_mean(values: Sequence[float | int]) -> float | None:
    """Return mean or None for an empty sequence."""

    return float(sum(float(v) for v in values) / len(values)) if values else None


def build_dataset_report(
    *,
    args: argparse.Namespace,
    source_examples: list[SourceExample] | None,
    transform_summary: dict[str, Any],
    split_summary: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build JSON metadata for the constructed dataset."""

    all_records = train_records + eval_records
    return {
        "created_at": now(),
        "config": serializable_config(args),
        "source": {
            "num_loaded": len(source_examples) if source_examples is not None else None,
            "input_path": str(args.input_path),
            "numeric_policy_filter": getattr(args, "source_filter_summary", {}),
        },
        "transformation": {
            "type": "numeric_literals_decimal_to_base_notation",
            "version": TRANSFORMATION_VERSION,
            "base": args.base,
            "max_decimal_places": args.max_decimal_places,
            "max_finite_fractional_digits": args.max_finite_fractional_digits,
            "decimal_policy": "source_max_places_then_finite_target_radix_else_exact_fraction",
            "teacher_decimal_policy": "no-source-decimal:finite-target-only; source-decimal:same-max-place-policy",
            "solution_fraction_style": "latex_frac",
            "problem_and_answer_fraction_style": "plain_slash",
            "require_exactly_one_teacher_boxed_answer": True,
            "reject_scientific": True,
            "normalize_comma_grouping": True,
            "convert_numeric_substrings_inside_identifiers": True,
            "drop_unchanged_final_answer": args.drop_unchanged_final_answer,
            "note": "Every numeric substring is transformed, including implicit coefficients and identifier/subscript digits. Default dataset columns are transformed/base-N; original columns are stored with original_* prefixes.",
        },
        "expert_generation": {
            "model": args.expert_model,
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.expert_num_generations,
            "temperature": args.expert_temperature,
            "top_p": args.expert_top_p,
        },
        "transform_summary": transform_summary,
        "split_summary": split_summary,
        "dataset_summary": {
            "total_rows": len(all_records),
            "train_rows": len(train_records),
            "eval_rows": len(eval_records),
            "module_counts": count_by_key(all_records, "module"),
            "difficulty_counts": count_by_key(all_records, "difficulty"),
            "bucket_counts": count_by_key(all_records, "bucket"),
            "output_token_quantiles": quantiles([float(r.get("output_tokens", -1)) for r in all_records if int(r.get("output_tokens", -1)) >= 0]),
        },
        "output_files": {
            "train_data": str(Path(args.output_root) / "train_data"),
            "eval_data": str(Path(args.output_root) / "eval_data"),
            "accepted_records": str(accepted_record_path(Path(args.output_root))),
            "expert_generation_records": str(generation_record_path(Path(args.output_root))),
            "dataset_report": str(Path(args.output_root) / "math_contradiction_report.json"),
            "preview": str(Path(args.output_root) / "dataset_preview.txt"),
        },
    }


def preview_value(value: Any, max_len: int = 1200) -> str:
    """Render one field for the dataset preview."""

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def write_dataset_preview(output_root: Path, train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], *, num_examples: int) -> None:
    """Write a text preview similar in spirit to visualize_data.py."""

    lines: list[str] = []

    def header(title: str) -> None:
        lines.append("\n" + "=" * 100)
        lines.append(title)
        lines.append("=" * 100)

    header("DATASET SUMMARY")
    lines.append(f"Output root: {output_root}")
    lines.append(f"Train rows: {len(train_records)}")
    lines.append(f"Eval rows: {len(eval_records)}")
    columns = sorted(train_records[0].keys()) if train_records else []
    lines.append("Columns:")
    for col in columns:
        lines.append(f"  - {col}")

    for split_name, records in (("train", train_records), ("eval", eval_records)):
        header(f"RAW EXAMPLES: {split_name}")
        for i, row in enumerate(records[:num_examples]):
            lines.append(f"\n--- Example {i} ---")
            important_keys = [
                "problem",
                "answer",
                "output_text",
                "original_problem",
                "original_answer",
                "original_output_text",
                "module",
                "difficulty",
                "source_id",
            ]
            for key in important_keys:
                if key in row:
                    lines.append(f"\n[{key}]")
                    lines.append(preview_value(row[key]))

    path = output_root / "dataset_preview.txt"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logging.info("Wrote dataset preview to %s", path)


def build_prompt_texts_for_eval(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    world: str,
    enable_thinking: bool | None,
) -> tuple[list[str], list[str]]:
    """Build prompts and references for original or mod-world evaluation."""

    if world not in {"original", "mod"}:
        raise ValueError(f"Unknown world: {world}")

    prompts: list[str] = []
    references: list[str] = []
    for record in records:
        if world == "original":
            problem = record["original_problem"]
            answer = record["original_answer"]
        else:
            problem = record["problem"]
            answer = record["answer"]
        prompts.append(
            apply_chat_template_compat(
                tokenizer,
                build_openr1_messages(problem),
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        references.append(str(answer))
    return prompts, references


def score_completion_groups(
    *,
    completions: list[VLLMCompletion],
    references: list[str],
    verifier: BaseAwareMathVerifier,
    world: str,
    n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score grouped n-completion outputs and compute acc1/pass@k."""

    grouped: dict[int, list[VLLMCompletion]] = defaultdict(list)
    for completion in completions:
        grouped[completion.row_idx].append(completion)

    row_records: list[dict[str, Any]] = []
    acc1_scores: list[int] = []
    acck_scores: list[int] = []
    hit_limit_flags: list[int] = []
    lengths: list[float] = []

    for row_idx in sorted(grouped):
        group = sorted(grouped[row_idx], key=lambda c: c.response_idx)
        ref = references[row_idx]
        scored_outputs: list[dict[str, Any]] = []
        scores: list[int] = []
        for completion in group:
            details = verifier.verify_with_details(
                ref,
                completion.text,
                world=world,
            )
            score = int(details.correct)
            scores.append(score)
            hit_limit_flags.append(int(completion.hit_token_limit))
            lengths.append(float(completion.generated_num_tokens))
            scored_outputs.append(
                {
                    "response_idx": completion.response_idx,
                    "response": completion.text,
                    "correct": bool(score),
                    "verify_method": details.method,
                    "verify_extracted_prediction": details.extracted_prediction,
                    "verify_error": details.error,
                    "verify_decimal_gold": normalize_text(getattr(details, "decimal_gold", "")),
                    "verify_decimal_prediction": normalize_text(getattr(details, "decimal_prediction", "")),
                    "generated_num_tokens": completion.generated_num_tokens,
                    "finish_reason": completion.finish_reason,
                    "hit_token_limit": completion.hit_token_limit,
                }
            )

        acc1 = scores[0] if scores else 0
        acck = int(any(scores[:n]))
        acc1_scores.append(acc1)
        acck_scores.append(acck)
        row_records.append(
            {
                "row_idx": row_idx,
                "reference": ref,
                "acc1_correct": bool(acc1),
                f"acc{n}_correct": bool(acck),
                "outputs": scored_outputs,
            }
        )

    summary = {
        "num_rows": len(row_records),
        "n": n,
        "acc1": safe_mean(acc1_scores) or 0.0,
        f"acc{n}": safe_mean(acck_scores) or 0.0,
        "num_acc1_correct": int(sum(acc1_scores)),
        f"num_acc{n}_correct": int(sum(acck_scores)),
        "hit_token_limit_ratio": safe_mean(hit_limit_flags) or 0.0,
        "length_tokens": {**quantiles(lengths), "mean": safe_mean(lengths)},
    }
    return row_records, summary


def evaluate_student_one_world_split(
    *,
    llm: Any,
    tokenizer: Any,
    scorer: Any,
    records: list[dict[str, Any]],
    output_dir: Path,
    split_name: str,
    world: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate acc0, acc1, and acc4 for one split/world."""

    prompts, references = build_prompt_texts_for_eval(
        records,
        tokenizer,
        world=world,
        enable_thinking=False if args.student_disable_thinking else None,
    )
    converter = BaseIntegerConverter(
        base=args.base,
        max_decimal_places=args.max_decimal_places,
        max_finite_fractional_digits=args.max_finite_fractional_digits,
    )
    verifier = BaseAwareMathVerifier(scorer, converter)

    if args.save_student_prompts:
        prompt_records = [
            {
                "row_idx": i,
                "source_id": records[i].get("source_id"),
                "world": world,
                "split": split_name,
                "prompt": prompts[i],
                "reference": references[i],
            }
            for i in range(len(prompts))
        ]
        save_json(output_dir / f"{split_name}_{world}_student_prompts.json", prompt_records)

    # acc0: deterministic greedy single generation.
    greedy_completions: list[VLLMCompletion] = []
    for batch_no, idxs in enumerate(chunks(list(range(len(prompts))), args.student_eval_batch_size), start=1):
        greedy_completions.extend(
            generate_with_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompts=[prompts[i] for i in idxs],
                row_indices=list(idxs),
                n=1,
                max_new_tokens=args.student_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=args.seed + 10_000 + batch_no,
                truncate_prompt_tokens=args.max_prompt_tokens,
            )
        )
    greedy_records, greedy_summary = score_completion_groups(
        completions=greedy_completions,
        references=references,
        verifier=verifier,
        world=world,
        n=1,
    )
    acc0_scores = [int(r["acc1_correct"]) for r in greedy_records]
    greedy_summary = {
        "acc0": safe_mean(acc0_scores) or 0.0,
        "num_acc0_correct": int(sum(acc0_scores)),
        "num_rows": len(greedy_records),
        "hit_token_limit_ratio": greedy_summary["hit_token_limit_ratio"],
        "length_tokens": greedy_summary["length_tokens"],
    }

    # acc1/acc4: four sampled generations; acc1 uses the first sample, acc4 is pass@4.
    sample_n = int(args.student_acc4_num_samples)
    sampled_completions: list[VLLMCompletion] = []
    for batch_no, idxs in enumerate(chunks(list(range(len(prompts))), args.student_eval_batch_size), start=1):
        sampled_completions.extend(
            generate_with_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompts=[prompts[i] for i in idxs],
                row_indices=list(idxs),
                n=sample_n,
                max_new_tokens=args.student_max_new_tokens,
                temperature=args.student_temperature,
                top_p=args.student_top_p,
                seed=args.seed + 20_000 + batch_no,
                truncate_prompt_tokens=args.max_prompt_tokens,
            )
        )
    sampled_records, sampled_summary = score_completion_groups(
        completions=sampled_completions,
        references=references,
        verifier=verifier,
        world=world,
        n=sample_n,
    )

    out_records: list[dict[str, Any]] = []
    sampled_by_idx = {int(r["row_idx"]): r for r in sampled_records}
    greedy_by_idx = {int(r["row_idx"]): r for r in greedy_records}
    for i, record in enumerate(records):
        out_records.append(
            {
                "row_idx": i,
                "source_id": record.get("source_id"),
                "problem_hash": record.get("problem_hash"),
                "module": record.get("module"),
                "difficulty": record.get("difficulty"),
                "bucket": record.get("bucket"),
                "split": split_name,
                "world": world,
                "reference": references[i],
                "greedy": greedy_by_idx.get(i),
                "sampled": sampled_by_idx.get(i),
            }
        )

    records_path = output_dir / f"{split_name}_{world}_student_eval_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    append_jsonl(records_path, out_records)

    summary = {
        "split": split_name,
        "world": world,
        "num_rows": len(records),
        "acc0": greedy_summary["acc0"],
        "acc1": sampled_summary["acc1"],
        "acc4": sampled_summary.get(f"acc{sample_n}"),
        "num_acc0_correct": greedy_summary["num_acc0_correct"],
        "num_acc1_correct": sampled_summary["num_acc1_correct"],
        "num_acc4_correct": sampled_summary.get(f"num_acc{sample_n}_correct"),
        "greedy_hit_token_limit_ratio": greedy_summary["hit_token_limit_ratio"],
        "sampled_hit_token_limit_ratio": sampled_summary["hit_token_limit_ratio"],
        "greedy_length_tokens": greedy_summary["length_tokens"],
        "sampled_length_tokens": sampled_summary["length_tokens"],
        "records_path": str(records_path),
    }
    logging.info(
        "Student eval %s/%s: acc0=%.4f acc1=%.4f acc4=%.4f rows=%d",
        split_name,
        world,
        float(summary["acc0"]),
        float(summary["acc1"]),
        float(summary["acc4"]),
        len(records),
    )
    return summary


def evaluate_student(
    *,
    output_root: Path,
    scorer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate the student on saved original and transformed splits."""

    output_dir = output_root / "student_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = None
    tokenizer = None
    start_time = time.time()
    try:
        llm, tokenizer = load_vllm_model(
            model_name=args.student_model,
            tokenizer_name=args.student_tokenizer,
            gpu_memory_utilization=args.student_gpu_memory_utilization,
            max_model_len=args.student_max_model_len,
            tensor_parallel_size=args.student_tensor_parallel_size,
            dtype=args.student_dtype,
            seed=args.seed,
        )

        summaries: dict[str, Any] = {}
        for split_name in args.student_eval_splits:
            records = load_built_split(output_root, split_name)
            if args.student_eval_limit is not None:
                records = records[: int(args.student_eval_limit)]
            for world in ("original", "mod"):
                key = f"{split_name}_{world}"
                summaries[key] = evaluate_student_one_world_split(
                    llm=llm,
                    tokenizer=tokenizer,
                    scorer=scorer,
                    records=records,
                    output_dir=output_dir,
                    split_name=split_name,
                    world=world,
                    args=args,
                )

        report = {
            "created_at": now(),
            "student_model": args.student_model,
            "student_max_new_tokens": args.student_max_new_tokens,
            "student_temperature": args.student_temperature,
            "student_top_p": args.student_top_p,
            "student_acc4_num_samples": args.student_acc4_num_samples,
            "splits": list(args.student_eval_splits),
            "summaries": summaries,
            "elapsed_seconds": time.time() - start_time,
        }
        save_json(output_dir / "student_eval_summary.json", report)
        return report
    finally:
        logging.info("Destroying student model.")
        destroy_vllm_engine(llm)
        try:
            del tokenizer
        except Exception:
            pass
        cleanup_cuda()


def build_markdown_report(report: dict[str, Any], student_report: dict[str, Any] | None) -> str:
    """Build a compact Markdown report for manual inspection."""

    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Math contradiction dataset report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Input path: `{cfg['input_path']}`")
    lines.append(f"- Output root: `{cfg['output_root']}`")
    lines.append(f"- Base: `{cfg['base']}`")
    lines.append(f"- Expert model: `{cfg['expert_model']}`")
    lines.append(f"- Expert max new tokens: `{cfg['max_new_tokens']}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    ds = report["dataset_summary"]
    lines.append(f"- Total rows: `{ds['total_rows']}`")
    lines.append(f"- Train rows: `{ds['train_rows']}`")
    lines.append(f"- Eval rows: `{ds['eval_rows']}`")
    lines.append(f"- Actual eval fraction: `{report['split_summary']['actual_eval_fraction']:.4f}`")
    lines.append("")
    lines.append("### Difficulty counts")
    lines.append("")
    for key, value in sorted(ds["difficulty_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("### Module counts")
    lines.append("")
    for key, value in sorted(ds["module_counts"].items()):
        lines.append(f"- `{key}`: {value}")

    if student_report:
        lines.append("")
        lines.append("## Student evaluation")
        lines.append("")
        lines.append("| split/world | rows | acc0 | acc1 | acc4 |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, summary in sorted(student_report.get("summaries", {}).items()):
            lines.append(
                f"| {key} | {summary['num_rows']} | {summary['acc0']:.4f} | "
                f"{summary['acc1']:.4f} | {summary['acc4']:.4f} |"
            )
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for key, path in report["output_files"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build a base-converted math contradiction dataset.")

    # Paths / execution modes.
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH, help="Directory/file containing filtered_math_dataset.jsonl or a HF dataset.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Where train_data/eval_data and metadata will be saved.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated split directories and intermediate transformed records.")
    parser.add_argument("--overwrite_splits", action="store_true", help="Overwrite only train_data/eval_data when saving splits.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing expert generation/transformed JSONL files.")
    parser.add_argument("--skip_expert_generation", action="store_true", help="Do not load/generate expert solutions; use existing expert_generation_records.jsonl.")
    parser.add_argument("--only_student_eval", action="store_true", help="Skip dataset construction and evaluate an already-saved dataset only.")
    parser.add_argument("--skip_student_eval", action="store_true", help="Build the dataset but do not evaluate the student.")
    parser.add_argument("--force_rebuild_transforms", action="store_true", help="Ignore accepted_transformed_records.jsonl and rebuild it from expert records.")

    # Dataset construction.
    parser.add_argument("--base", type=int, default=9, help="Target numeral base for the contradiction world.")
    parser.add_argument(
        "--max_decimal_places",
        type=int,
        default=2,
        help="Maximum decimal places allowed in original questions/answers and in teacher responses to decimal-containing questions.",
    )
    parser.add_argument(
        "--max_finite_fractional_digits",
        type=int,
        default=12,
        help="Maximum target-base fractional digits used before falling back to an exact fraction.",
    )
    parser.add_argument(
        "--drop_unchanged_final_answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop rows whose compact answer string is unchanged by conversion. "
            "Disabled by default because the transformed problem may still require base-N reasoning; "
            "the decimal-consistency filter handles genuinely invariant tasks."
        ),
    )
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_source_rows", type=int, default=None, help="Optional limit before expert generation, useful for debug.")
    parser.add_argument("--max_final_rows", type=int, default=None, help="Optional limit after verification/transformation.")
    parser.add_argument(
        "--drop_decimal_consistent_mod",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop transformed examples whose ordinary-decimal solution already equals the transformed target answer.",
    )
    parser.add_argument(
        "--drop_degenerate_identity_patterns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop obvious identity/annihilator cases such as division by 1 and multiplication by 0/1.",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--preview_examples", type=int, default=15)

    # Expert generation.
    parser.add_argument("--expert_model", type=str, default=DEFAULT_EXPERT_MODEL)
    parser.add_argument("--expert_tokenizer", type=str, default=None)
    parser.add_argument("--expert_num_generations", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Expert max new tokens. Also matches your requested dataset default.")
    parser.add_argument("--expert_temperature", type=float, default=0.2)
    parser.add_argument("--expert_top_p", type=float, default=0.95)
    parser.add_argument("--expert_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--expert_max_model_len", type=int, default=2048)
    parser.add_argument("--expert_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--expert_dtype", type=str, default="bfloat16")
    parser.add_argument("--expert_disable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation_batch_size", type=int, default=256)

    # Student eval.
    parser.add_argument("--student_model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--student_tokenizer", type=str, default=None)
    parser.add_argument("--student_max_new_tokens", type=int, default=512)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--student_top_p", type=float, default=0.95)
    parser.add_argument("--student_acc4_num_samples", type=int, default=4)
    parser.add_argument("--student_gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--student_max_model_len", type=int, default=2048)
    parser.add_argument("--student_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--student_dtype", type=str, default="bfloat16")
    parser.add_argument("--student_disable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--student_eval_batch_size", type=int, default=256)
    parser.add_argument("--student_eval_splits", nargs="+", choices=("train", "eval"), default=["eval"], help="Splits to evaluate. Default is eval only.")
    parser.add_argument("--student_eval_limit", type=int, default=None)
    parser.add_argument("--save_student_prompts", action="store_true")

    # Prompt/model length.
    parser.add_argument("--max_prompt_tokens", type=int, default=1536)

    # Logging/test.
    parser.add_argument("--log_level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--self_test", action="store_true", help="Run lightweight tests and exit.")

    args = parser.parse_args()

    if args.expert_num_generations < 1:
        raise ValueError("--expert_num_generations must be >= 1")
    if args.expert_num_generations > 1 and args.expert_temperature <= 0:
        raise ValueError(
            "--expert_num_generations > 1 requires --expert_temperature > 0; "
            "otherwise vLLM produces duplicate greedy candidates"
        )
    if args.generation_batch_size < 1 or args.student_eval_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.student_acc4_num_samples != 4:
        raise ValueError("This script reports acc4, so --student_acc4_num_samples must remain 4")
    if args.max_new_tokens <= 0 or args.student_max_new_tokens <= 0:
        raise ValueError("max_new_tokens values must be positive")
    if args.max_decimal_places < 0:
        raise ValueError("--max_decimal_places must be >= 0")
    if args.max_finite_fractional_digits < 1:
        raise ValueError("--max_finite_fractional_digits must be >= 1")
    if not 0.0 < args.eval_fraction < 1.0:
        raise ValueError("--eval_fraction must be in (0, 1)")
    return args


def run_self_test() -> None:
    """Run lightweight tests that do not require vLLM or project imports."""

    conv = BaseIntegerConverter(base=9, max_decimal_places=2)
    assert conv.direct_number(57) == "63"
    assert conv.direct_number(18) == "20"
    assert conv.direct_number(75) == "83"

    original = r"57 + 18 = 75, so \boxed{75}."
    mod = conv.direct_text(original)
    assert mod == r"63 + 20 = 83, so \boxed{83}.", mod
    assert conv.reverse_text(mod) == original
    assert conv.direct_text("10/3") == "11/3"

    # Grouping separators are normalized, not rejected.
    assert conv.direct_text("9,486,900 / 10,541") == "18758520 / 15412"

    # 3.14 is exact 157/50, represented as 184/55 in base 9 notation.
    assert conv.direct_text("3.14") == "(184/55)"
    assert conv.reverse_text("(184/55)") == "(157/50)"

    # Fractions terminating in the target base remain radix literals.
    base8 = BaseIntegerConverter(base=8, max_decimal_places=2)
    assert base8.direct_text("0.5") == "0.4"
    assert base8.reverse_text("0.4") == "(1/2)"

    try:
        conv.direct_text("1.2345")
        raise AssertionError("decimals longer than the configured limit should be rejected")
    except UnsupportedNumericNotationError:
        pass
    try:
        conv.reverse_text("9")
        raise AssertionError("digit 9 must be invalid in base 9")
    except UnsupportedNumericNotationError:
        pass

    decimal_answer, method = solve_problem_under_decimal(
        "Calculate 107 divided by 1.", "arithmetic__div"
    )
    assert decimal_answer == "107" and method == "sympy_arithmetic_decimal", (
        decimal_answer,
        method,
    )
    decimal_answer, method = solve_problem_under_decimal(
        "(31 - (-25 + 25)) + 1 + 54", "arithmetic__add_sub_multiple"
    )
    assert decimal_answer == "86" and method == "sympy_arithmetic_decimal", (
        decimal_answer,
        method,
    )
    decimal_answer, method = solve_problem_under_decimal(
        "Solve 4*c = -4*u + c + 48, 30 = 22*c + 11*c + 30 for u.",
        "algebra__linear_2d",
    )
    assert decimal_answer == "12" and method == "sympy_linear_2d_decimal", (
        decimal_answer,
        method,
    )

    # Test the base-aware adapter with a tiny fake decimal scorer.
    @dataclass(frozen=True)
    class FakeDetails:
        correct: bool
        method: str = "fake"
        extracted_prediction: str = ""
        gold_parsed: str = ""
        prediction_parsed: str = ""
        error: str = ""

    class FakeScorer:
        def verify_with_details(self, gold: Any, prediction: Any, **_: Any) -> FakeDetails:
            boxed = extract_last_boxed(prediction)
            return FakeDetails(correct=str(gold) == str(boxed), extracted_prediction=str(boxed or ""))

    verifier = BaseAwareMathVerifier(FakeScorer(), conv)
    details = verifier.verify_with_details("13", r"Reasoning. \boxed{13}", world="mod")
    assert details.correct and details.decimal_gold == "12" and details.decimal_prediction == "12"
    details = verifier.verify_with_details("13", r"Reasoning. \boxed{19}", world="mod")
    assert not details.correct and "invalid for base 9" in details.error

    records = [
        {
            "bucket": "a/easy",
            "problem_hash": f"a{i}",
            "answer": "11",
            "original_answer": "10",
            "problem": "p",
            "mod_problem": "p",
            "mod_answer": "11",
            "messages": [],
            "output_text": "",
            "visible_output_text": "",
            "golden_answer": "11",
            "golden_response": "",
            "original_problem": "",
            "original_output_text": "",
        }
        for i in range(20)
    ] + [
        {
            "bucket": "b/hard",
            "problem_hash": f"b{i}",
            "answer": "11",
            "original_answer": "10",
            "problem": "p",
            "mod_problem": "p",
            "mod_answer": "11",
            "messages": [],
            "output_text": "",
            "visible_output_text": "",
            "golden_answer": "11",
            "golden_response": "",
            "original_problem": "",
            "original_output_text": "",
        }
        for i in range(20)
    ]
    train, eval_records, summary = stratified_train_eval_split(
        records, eval_fraction=0.1, seed=1
    )
    assert len(train) == 36 and len(eval_records) == 4, summary
    validate_final_records(train, eval_records)
    # Source decimals use at most two places and choose output formatting by field.
    assert conv.direct_text(
        "Compute 3.14 + 1.",
        fraction_style="plain",
        parenthesize_plain_fractions=True,
    ) == "Compute (184/55) + 1."
    assert conv.direct_text(
        r"Thus 3.14 is exact.",
        fraction_style="latex",
        parenthesize_plain_fractions=False,
    ) == r"Thus \frac{184}{55} is exact."

    # Structural numbering is not a mathematical decimal.
    assert conv.direct_text("Step 2.1: compute 10.") == "Step 2.1: compute 11."

    # Every numeric substring is transformed, including implicit coefficients,
    # identifiers, and subscripts. This prevents mixed-base reasoning such as
    # leaving ``595h`` in decimal while converting standalone coefficients.
    connected = r"595h + x10 + item2 + abc_3 + x_{10} + 82p"
    connected_mod = r"731h + x11 + item2 + abc_3 + x_{11} + 101p"
    assert conv.direct_text(connected) == connected_mod
    assert conv.reverse_text(connected_mod) == connected
    assert conv.direct_text("-50999y + 50997y") == "-76855y + 76853y"
    assert conv.direct_text(
        "x3.14y",
        fraction_style="plain",
        parenthesize_plain_fractions=True,
    ) == "x(184/55)y"

    # Teacher approximations in integer-only questions are rejected in base 9.
    candidate = {
        "correct": True,
        "hit_token_limit": False,
        "question": "Divide 5 by 2546.",
        "answer": "5/2546",
        "visible_response": r"Exact: \boxed{\frac{5}{2546}} and 0.00196386.",
    }
    valid, reason = validate_teacher_candidate(candidate, converter=conv)
    assert not valid and reason.startswith("teacher_introduced_nonfinite_decimal"), reason

    # Exactly one box is mandatory, even when both boxes are correct.
    candidate["visible_response"] = (
        r"\boxed{\frac{5}{2546}} or \boxed{\frac{5}{2546}}"
    )
    valid, reason = validate_teacher_candidate(candidate, converter=conv)
    assert not valid and reason == "boxed_answer_count:2", reason

    # A single exact response is accepted.
    candidate["visible_response"] = r"Therefore \boxed{\frac{5}{2546}}."
    valid, reason = validate_teacher_candidate(candidate, converter=conv)
    assert valid and reason == "accepted", reason

    print("self_test passed")


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    input_path = Path(args.input_path).expanduser()
    output_root = Path(args.output_root).expanduser()
    configure_logging(output_root, args.log_level)
    logging.info("Starting math_contradiction dataset build")
    logging.info("Config: %s", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))

    ensure_project_imports(Path(args.project_root))
    from data_utils.math_verify_scorer import OpenR1MathVerifyScorer, extract_visible_response  # type: ignore

    # Store the function on args so generation code can use the exact project helper
    # without re-importing in every inner loop.
    args.extract_visible_response = extract_visible_response
    scorer = OpenR1MathVerifyScorer()

    source_examples: list[SourceExample] | None = None
    transform_summary: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    if not args.only_student_eval:
        loaded_source_examples = load_source_examples(input_path, limit=args.max_source_rows)
        source_policy_converter = BaseIntegerConverter(
            base=args.base,
            max_decimal_places=args.max_decimal_places,
            max_finite_fractional_digits=args.max_finite_fractional_digits,
        )
        source_examples, source_filter_summary = filter_source_examples_for_numeric_policy(
            loaded_source_examples,
            converter=source_policy_converter,
        )
        args.source_filter_summary = source_filter_summary
        logging.info(
            "Source numeric-policy filter kept %d/%d rows",
            len(source_examples),
            len(loaded_source_examples),
        )
        if source_filter_summary.get("rejection_counts"):
            logging.info(
                "Source numeric-policy rejection counts: %s",
                source_filter_summary["rejection_counts"],
            )
        save_json(output_root / "run_config.json", serializable_config(args))

        if output_root.exists() and args.overwrite:
            # Avoid deleting input data if user sets input_path == output_root.
            for child in (output_root / "train_data", output_root / "eval_data"):
                if child.exists():
                    shutil.rmtree(child)
            for file_name in ("accepted_transformed_records.jsonl", "math_contradiction_report.json", "math_contradiction_report.md", "dataset_preview.txt"):
                p = output_root / file_name
                if p.exists():
                    p.unlink()
            if not args.resume:
                gen_path = generation_record_path(output_root)
                if gen_path.exists():
                    gen_path.unlink()

        if not args.skip_expert_generation:
            generation_summary = generate_expert_demonstrations(
                source_examples=source_examples,
                scorer=scorer,
                output_root=output_root,
                args=args,
            )
        else:
            generation_summary = {
                "skipped": True,
                "records_path": str(generation_record_path(output_root)),
                "existing_records": len(read_jsonl(generation_record_path(output_root))),
            }
            logging.info("Skipping expert generation; using existing records at %s", generation_record_path(output_root))

        transform_summary = build_transformed_records(output_root=output_root, scorer=scorer, args=args)
        transform_summary["generation_summary"] = generation_summary

        accepted_records = read_jsonl(accepted_record_path(output_root))
        if not accepted_records:
            raise RuntimeError("No accepted transformed records. Check expert generation accuracy and numeric filters.")

        train_records, eval_records, split_summary = stratified_train_eval_split(
            accepted_records,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        validate_final_records(
            train_records,
            eval_records,
            require_changed_final_answer=bool(args.drop_unchanged_final_answer),
        )
        save_splits(
            output_root,
            train_records,
            eval_records,
            overwrite_splits=bool(args.overwrite or args.overwrite_splits or args.resume),
        )
        write_dataset_preview(
            output_root,
            train_records,
            eval_records,
            num_examples=args.preview_examples,
        )

        report = build_dataset_report(
            args=args,
            source_examples=source_examples,
            transform_summary=transform_summary,
            split_summary=split_summary,
            train_records=train_records,
            eval_records=eval_records,
        )
        save_json(output_root / "math_contradiction_report.json", report)
    else:
        logging.info("Only student evaluation requested; skipping dataset construction")
        train_records = load_built_split(output_root, "train") if (output_root / "train_data").exists() else []
        eval_records = load_built_split(output_root, "eval") if (output_root / "eval_data").exists() else []
        report_path = output_root / "math_contradiction_report.json"
        report = read_json(report_path) if report_path.exists() else {
            "created_at": now(),
            "config": serializable_config(args),
            "dataset_summary": {"train_rows": len(train_records), "eval_rows": len(eval_records), "total_rows": len(train_records) + len(eval_records)},
            "output_files": {},
        }

    student_report: dict[str, Any] | None = None
    if not args.skip_student_eval:
        student_report = evaluate_student(output_root=output_root, scorer=scorer, args=args)
        save_json(output_root / "student_eval_summary.json", student_report)
    else:
        logging.info("Skipping student evaluation")

    markdown = build_markdown_report(report, student_report)
    (output_root / "math_contradiction_report.md").write_text(markdown, encoding="utf-8")
    logging.info("Done. Report saved to %s", output_root / "math_contradiction_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
