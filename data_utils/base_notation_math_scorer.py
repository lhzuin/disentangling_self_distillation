from __future__ import annotations

"""Exact scoring helpers for math written in a non-decimal positional base.

The model-facing prompt and response stay in the configured base. For scoring,
only the compact gold answer and the final ``\\boxed{...}`` prediction are
rewritten to exact decimal notation and delegated to the existing
Math-Verify scorer.

This module intentionally contains no dataset-building or model-generation code,
so it is safe to import from training and evaluation paths.
"""

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Iterator


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


class UnsupportedNumericNotationError(ValueError):
    """Raised when a numeric span cannot be back-converted exactly."""


@dataclass(frozen=True)
class BaseNotationConverter:
    """Rewrite numeric literals from ``base`` notation to exact decimal text.

    Every numeric substring is interpreted in the configured base, including
    digits attached to letters. Structural outline labels such as ``Step 2.1:``
    are the only exception, matching the dataset-construction policy.
    """

    base: int = 9
    reject_scientific: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.base <= 10:
            raise ValueError("base must be in [2, 10]")

    @staticmethod
    def _digits() -> str:
        return "0123456789"

    @staticmethod
    def _strip_grouping(token: str) -> str:
        return token.replace(",", "")

    @staticmethod
    def _is_structural_decimal_match(text: str, start: int, end: int) -> bool:
        prefix = text[max(0, start - 24) : start]
        suffix = text[end : end + 4]
        return bool(
            re.search(r"(?:step|section|part|case)\s*$", prefix, flags=re.IGNORECASE)
            and re.match(r"\s*[:.)]", suffix)
        )

    def validate_supported_text(self, text: Any, *, label: str = "text") -> None:
        source = str(text or "")
        if self.reject_scientific and SCIENTIFIC_NUMBER_RE.search(source):
            raise UnsupportedNumericNotationError(
                f"{label} contains unsupported scientific notation"
            )

    def reverse_integer(self, token: str) -> int:
        normalized = self._strip_grouping(token).strip()
        sign = -1 if normalized.startswith("-") else 1
        unsigned = normalized.lstrip("+-")
        if not unsigned:
            raise ValueError(f"Invalid integer token: {token!r}")

        valid_digits = self._digits()[: self.base]
        value = 0
        for char in unsigned:
            if char not in valid_digits:
                raise ValueError(
                    f"Digit {char!r} is invalid for base {self.base}: {token!r}"
                )
            value = value * self.base + valid_digits.index(char)
        return sign * value

    def _reverse_radix_token(self, token: str) -> str:
        normalized = self._strip_grouping(token)
        sign = -1 if normalized.startswith("-") else 1
        unsigned = normalized.lstrip("+-")

        if "." not in unsigned:
            return str(self.reverse_integer(normalized))

        integer_part, fractional_part = unsigned.split(".", 1)
        if not fractional_part:
            integer = self.reverse_integer(("-" if sign < 0 else "") + (integer_part or "0"))
            return str(integer)

        valid_digits = self._digits()[: self.base]
        digits = (integer_part or "0") + fractional_part
        invalid = [char for char in digits if char not in valid_digits]
        if invalid:
            raise ValueError(
                f"Digit {invalid[0]!r} is invalid for base {self.base}: {token!r}"
            )

        integer_value = self.reverse_integer(integer_part or "0")
        fractional_value = 0
        for char in fractional_part:
            fractional_value = fractional_value * self.base + valid_digits.index(char)

        value = Fraction(integer_value, 1) + Fraction(
            fractional_value,
            self.base ** len(fractional_part),
        )
        value *= sign
        if value.denominator == 1:
            return str(value.numerator)
        return f"({value.numerator}/{value.denominator})"

    def iter_numeric_tokens(self, text: Any) -> Iterator[tuple[str, int, int]]:
        source = str(text or "")
        for match in NUMBER_TOKEN_RE.finditer(source):
            start, end = match.span()
            if self._is_structural_decimal_match(source, start, end):
                continue
            yield match.group(0), start, end

    def reverse_text(self, text: Any, *, label: str = "text") -> str:
        """Back-convert every numeric substring to exact decimal notation."""

        source = str(text or "")
        self.validate_supported_text(source, label=label)

        def replace(match: re.Match[str]) -> str:
            start, end = match.span()
            if self._is_structural_decimal_match(source, start, end):
                return match.group(0)
            token = match.group(0)
            try:
                return self._reverse_radix_token(token)
            except Exception as exc:
                raise UnsupportedNumericNotationError(
                    f"Could not convert token {token!r} in {label}: {exc}"
                ) from exc

        return NUMBER_TOKEN_RE.sub(replace, source)


@dataclass(frozen=True)
class BaseAwareVerificationResult:
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


def _replace_latex_fractions(expression: str) -> str:
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
    """Parse a compact numeric expression into an exact SymPy rational."""

    raw = str(value or "").strip()
    if not raw:
        return None

    try:
        import sympy as sp

        text = raw.replace(r"\left", "").replace(r"\right", "")
        text = text.replace(r"\,", "").replace(",", "")
        text = text.replace(r"\cdot", "*").replace(r"\times", "*")
        text = text.replace("−", "-").strip().strip("$")
        for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
        if "=" in text:
            text = text.rsplit("=", 1)[1].strip()

        text = _replace_latex_fractions(text)
        text = text.replace("{", "(").replace("}", ")").replace("^", "**")
        if re.search(r"[^0-9+\-*/().\s]", text):
            return None

        parsed = sp.sympify(text, rational=True)
        if getattr(parsed, "free_symbols", set()):
            return None
        simplified = sp.cancel(parsed)
        return simplified if simplified.is_Rational else None
    except Exception:
        return None


def visible_response_for_scoring(text: Any) -> tuple[str, str | None]:
    raw = str(text or "")
    has_open = bool(re.search(r"<think\b[^>]*>", raw, flags=re.IGNORECASE))
    has_close = bool(re.search(r"</think\s*>", raw, flags=re.IGNORECASE))
    if has_open and not has_close:
        return "", "unclosed think block"
    return _THINK_BLOCK_RE.sub("", raw).strip(), None


class BaseAwareMathVerifier:
    """Score final boxed answers after exact base-to-decimal back-conversion."""

    def __init__(self, scorer: Any, *, base: int = 9) -> None:
        self.scorer = scorer
        self.converter = BaseNotationConverter(base=base)

    @property
    def base(self) -> int:
        return self.converter.base

    def verify_with_details(self, gold: Any, prediction: Any) -> BaseAwareVerificationResult:
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
            decimal_gold = self.converter.reverse_text(gold, label="base_gold")
            decimal_prediction = self.converter.reverse_text(
                boxed,
                label="base_boxed_prediction",
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
            method = f"base{self.base}_backconvert:strict_exact_numeric"
        else:
            correct = bool(delegated.correct)
            method = f"base{self.base}_backconvert:{delegated.method}"

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

    def score_many(self, responses: list[str], references: list[Any]) -> list[int]:
        if len(responses) != len(references):
            raise ValueError(
                f"responses/references length mismatch: {len(responses)} != {len(references)}"
            )
        return [
            int(self.verify_with_details(reference, response).correct)
            for response, reference in zip(responses, references)
        ]
