from __future__ import annotations

"""
Shared OpenR1-Math verifier for dataset construction and evaluation.

Policy
------
1. Try the public Open-R1 / TRL-style Math-Verify path first:
   - parse the gold answer with math_verify.parse(..., extraction_mode="first_match");
   - parse the full model response with LatexExtractionConfig using boxed priority,
     Open-R1 normalization options, and try_extract_without_anchor=False;
   - verify with math_verify.verify.
2. If the official full-response path fails, optionally try a narrow fallback:
   - extract only the last balanced \boxed{...} block;
   - parse that extracted final answer with Math-Verify;
   - verify against the same parsed gold answer.
3. Do not use natural-language answer-phrase regexes. Dataset construction should
   not accept examples that are only recoverable from phrases like "answer is ..."
   if the boxed answer cannot be verified.

The math-contradiction construction and evaluation paths both use this class,
directly or through ``BaseAwareMathVerifier``.

That keeps selected gold responses and later model responses on the same local
public verification implementation.
"""

import inspect
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    method: str = ""
    extracted_prediction: str = ""
    gold_parsed: str = ""
    prediction_parsed: str = ""
    error: str = ""


@dataclass(frozen=True)
class VisibleResponse:
    text: str
    had_think_block: bool
    has_unclosed_think: bool
    num_think_blocks_removed: int


_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_visible_response(text: Any) -> VisibleResponse:
    """
    Return the user-facing response after removing closed <think>...</think> blocks.

    Correctness should be judged on the visible response, not on hidden thinking.
    If an output opens <think> but never closes it, we treat it as structurally
    invalid because no clear user-facing answer was produced.
    """
    raw = "" if text is None else str(text)
    had_open = bool(re.search(r"<think\b[^>]*>", raw, flags=re.IGNORECASE))
    had_close = bool(re.search(r"</think\s*>", raw, flags=re.IGNORECASE))

    removed_text, num_removed = _THINK_BLOCK_RE.subn("", raw)
    has_unclosed = had_open and not had_close

    return VisibleResponse(
        text=removed_text.strip(),
        had_think_block=had_open or had_close,
        has_unclosed_think=has_unclosed,
        num_think_blocks_removed=num_removed,
    )

def extract_last_boxed(text: Any) -> str | None:
    """
    Extract the content of the last balanced \boxed{...} block.

    Regex-only extraction is fragile for nested braces, e.g. \boxed{\frac{1}{2}}.
    This function scans braces and returns the last complete boxed expression.
    """
    text = str(text or "")
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    last_value: str | None = None

    for start in starts:
        brace_start = text.find("{", start)
        if brace_start < 0:
            continue

        depth = 0
        for pos in range(brace_start, len(text)):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_value = text[brace_start + 1 : pos].strip()
                    break

    return last_value


def _latex_env_candidates(expr: Any) -> list[str]:
    """Return conservative Math-Verify parse candidates for a final expression."""
    expr = str(expr or "").strip()
    if not expr:
        return []

    candidates = [expr]
    if "$" not in expr and r"\(" not in expr and r"\[" not in expr:
        candidates.extend([f"${expr}$", rf"\({expr}\)", rf"\boxed{{{expr}}}"])

    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


class OpenR1MathVerifyScorer:
    """
    Open-R1-style public Math-Verify scorer with a narrow last-boxed fallback.

    The class intentionally exposes both score_one(prediction, gold), which is
    convenient for DatasetAdapter.score_responses, and verify_with_details(gold,
    prediction), which is convenient for dataset construction diagnostics.
    """

    def __init__(self) -> None:
        try:
            from math_verify import LatexExtractionConfig, parse, verify  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenR1MathVerifyScorer requires math-verify. Install it with:\n"
                "  pip install 'math-verify[antlr4_13_2]'\n"
                "For best OpenR1 reproducibility, prefer math-verify==0.5.2 when possible."
            ) from exc

        try:
            from latex2sympy2_extended import NormalizationConfig  # type: ignore
        except Exception:
            NormalizationConfig = None  # type: ignore

        self.LatexExtractionConfig = LatexExtractionConfig
        self.NormalizationConfig = NormalizationConfig
        self.parse = parse
        self.verify = verify

    # ------------------------------------------------------------------
    # Version-compatible Math-Verify helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call a function while dropping kwargs unsupported by older versions."""
        try:
            return func(*args, **kwargs)
        except TypeError:
            try:
                signature = inspect.signature(func)
                supported = {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }
                return func(*args, **supported)
            except Exception:
                # Last resort: caller will handle the exception if even args fail.
                return func(*args)

    def _normalization_config(self) -> Any:
        """Build Open-R1 reward-style normalization config when available."""
        if self.NormalizationConfig is None:
            return {
                "nits": False,
                "malformed_operators": False,
                "basic_latex": True,
                "boxed": "all",
                "units": True,
            }

        kwargs = {
            "nits": False,
            "malformed_operators": False,
            "basic_latex": True,
            "boxed": "all",
            "units": True,
        }
        try:
            return self.NormalizationConfig(**kwargs)
        except TypeError:
            # Older NormalizationConfig variants may accept fewer fields.
            return self._call_with_supported_kwargs(self.NormalizationConfig, **kwargs)

    def _prediction_latex_config(self, *, official: bool) -> Any:
        """
        Build prediction extraction config.

        official=True mirrors Open-R1's public accuracy_reward as closely as the
        installed math-verify version permits. official=False is used only for
        the last-boxed fallback, where the input is already the extracted final
        expression, so requiring an answer anchor would be counterproductive.
        """
        kwargs = {
            "normalization_config": self._normalization_config(),
            "boxed_match_priority": 0,
        }
        if official:
            kwargs["try_extract_without_anchor"] = False
        else:
            kwargs["try_extract_without_anchor"] = True

        try:
            return self.LatexExtractionConfig(**kwargs)
        except TypeError:
            return self._call_with_supported_kwargs(self.LatexExtractionConfig, **kwargs)

    def _parse_gold(self, gold: Any) -> list[Any]:
        gold_text = "" if gold is None else str(gold)
        try:
            parsed = self._call_with_supported_kwargs(
                self.parse,
                gold_text,
                extraction_mode="first_match",
            )
            if parsed:
                return parsed
        except Exception:
            pass

        # Conservative fallback for plain numeric/LaTeX gold answers.
        for candidate in _latex_env_candidates(gold_text):
            try:
                parsed = self._call_with_supported_kwargs(
                    self.parse,
                    candidate,
                    extraction_mode="first_match",
                )
                if parsed:
                    return parsed
            except Exception:
                continue

        return []

    def _parse_prediction_official(self, prediction: Any) -> list[Any]:
        prediction_text = "" if prediction is None else str(prediction)
        try:
            return self._call_with_supported_kwargs(
                self.parse,
                prediction_text,
                extraction_config=[self._prediction_latex_config(official=True)],
                extraction_mode="first_match",
            )
        except Exception:
            return []

    def _parse_prediction_last_boxed(self, boxed_expr: str) -> tuple[list[Any], str]:
        for candidate in _latex_env_candidates(boxed_expr):
            try:
                parsed = self._call_with_supported_kwargs(
                    self.parse,
                    candidate,
                    extraction_config=[self._prediction_latex_config(official=False)],
                    extraction_mode="first_match",
                )
                if parsed:
                    return parsed, candidate
            except Exception:
                continue

            try:
                parsed = self._call_with_supported_kwargs(
                    self.parse,
                    candidate,
                    extraction_mode="first_match",
                )
                if parsed:
                    return parsed, candidate
            except Exception:
                continue

        return [], ""

    def _verify(self, gold_parsed: list[Any], pred_parsed: list[Any]) -> tuple[bool, str]:
        try:
            return bool(self.verify(gold_parsed, pred_parsed)), ""
        except Exception as exc:
            # Some math-verify versions expose timeout kwargs; use them only if supported.
            try:
                result = self._call_with_supported_kwargs(
                    self.verify,
                    gold_parsed,
                    pred_parsed,
                    timeout_seconds=0,
                )
                return bool(result), ""
            except Exception as exc2:
                return False, f"{type(exc).__name__}: {exc}; fallback: {type(exc2).__name__}: {exc2}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_with_details(
        self,
        gold_answer: Any,
        prediction: Any,
        *,
        allow_last_boxed_fallback: bool = True,
        strip_thinking: bool = True,
        require_visible_after_think: bool = False,
    ) -> VerificationResult:
        gold_parsed = self._parse_gold(gold_answer)
        if not gold_parsed:
            return VerificationResult(
                correct=False,
                method="gold_parse_failed",
                error=f"Could not parse gold answer: {gold_answer!r}",
            )
        
        prediction_for_scoring = prediction

        if strip_thinking:
            visible = extract_visible_response(prediction)
            if visible.has_unclosed_think:
                return VerificationResult(
                    correct=False,
                    method="unclosed_think_block",
                    gold_parsed=repr(gold_parsed),
                    error="Prediction contains an opening <think> without a closing </think>.",
                )

            if require_visible_after_think and visible.had_think_block and not visible.text:
                return VerificationResult(
                    correct=False,
                    method="empty_visible_response",
                    gold_parsed=repr(gold_parsed),
                    error="Prediction has a think block but no visible response after removing it.",
                )

            prediction_for_scoring = visible.text

        pred_parsed = self._parse_prediction_official(prediction_for_scoring)
        if pred_parsed:
            ok, err = self._verify(gold_parsed, pred_parsed)
            if ok:
                return VerificationResult(
                    correct=True,
                    method="official_full_response",
                    extracted_prediction=repr(pred_parsed),
                    gold_parsed=repr(gold_parsed),
                    prediction_parsed=repr(pred_parsed),
                )
            last_error = err
        else:
            last_error = "official prediction parse failed"

        if allow_last_boxed_fallback:
            boxed = extract_last_boxed(prediction_for_scoring)
            if boxed:
                pred_boxed_parsed, parsed_from = self._parse_prediction_last_boxed(boxed)
                if pred_boxed_parsed:
                    ok, err = self._verify(gold_parsed, pred_boxed_parsed)
                    if ok:
                        return VerificationResult(
                            correct=True,
                            method="last_boxed_fallback",
                            extracted_prediction=parsed_from or boxed,
                            gold_parsed=repr(gold_parsed),
                            prediction_parsed=repr(pred_boxed_parsed),
                        )
                    last_error = err or "last_boxed verify failed"
                else:
                    last_error = "last_boxed parse failed"
            else:
                last_error = "no boxed answer found"

        return VerificationResult(
            correct=False,
            method="failed",
            gold_parsed=repr(gold_parsed),
            error=last_error,
        )

    def is_correct(self, gold_answer: Any, prediction: Any) -> bool:
        return self.verify_with_details(
            gold_answer,
            prediction,
            strip_thinking=True,
            require_visible_after_think=False,
        ).correct

    def score_one(self, prediction: Any, gold_answer: Any) -> int:
        """Adapter-friendly argument order: prediction first, gold second."""
        return int(self.is_correct(gold_answer, prediction))

    def score_many(self, predictions: list[Any], gold_answers: list[Any]) -> list[int]:
        return [
            self.score_one(prediction, gold)
            for prediction, gold in zip(predictions, gold_answers)
        ]
