# contextualization_prompt_templates.py
"""Prompt templates used by ``contextualization_strategies.py``.

This module contains prompt text only. Strategy behavior, validation, dataset
restrictions, and registry entries live in ``contextualization_strategies.py``.
Templates use ``string.Template`` variables such as ``$question``,
``$compact_answer``, ``$full_reference``, ``$short_hint``,
``$student_response``, ``$feedback_text``, and ``$sibling_text``.

Variation ordering is stable: variation 1 is the canonical/default wording for
a strategy, while additional entries provide controlled prompt variants.
"""

from __future__ import annotations

# =============================================================================
# Static / precomputable strategies
# =============================================================================

NO_CONTEXT_PROMPT_TEMPLATES = [
    """$question""",
]


STATIC_REASONING_PROMPT_TEMPLATES = [
    """
$question

Before answering, carefully identify the required output format and the key constraints of the task.
Avoid changing the format requested by the problem.
If tool calls/actions are needed, make sure that both the action names and their inputs are exact.

Now answer the original question.
""",
    """
$question

First, check the required answer format and any constraints in the problem.
Then solve the task carefully, making sure not to change the expected output structure.

Now provide your answer.
""",
]

# GOLD_ANSWER_AS_EXAMPLE2_PROMPT_TEMPLATES = [
#     """
# $question

# This is an example for a final answer to the question:
# $compact_answer

# Now answer with a response of your own, including the thinking process.
# """,
# ]

GOLD_ANSWER_AS_EXAMPLE2_PROMPT_TEMPLATES = [
    """
$question

This is an example for a valid final answer to this question:
$compact_answer

Now answer with a response of your own, including the thinking process.
""",
]



GOLD_ANSWER_AS_EXAMPLE_PROMPT_TEMPLATES = [
    """
$question

This is an example for a response to the question:
$compact_answer

Now answer with a response of your own, including the thinking process.
""",
]


GOLD_ANSWER_AS_GUIDANCE_PROMPT_TEMPLATES = [
    """
$question

Helpful guidance for solving this instance:
$compact_answer

Use the guidance only to improve your answer. Do not simply copy it blindly.
Now answer the original question in the required format.
""",
]

GOLD_RESPONSE_AS_EXAMPLE_PROMPT_TEMPLATES = [
    """
$question

This is an example for a response to the question:
$full_reference

Now answer with a response of your own, including the thinking process.
""",
    """
$question

Here is a reference-style response for this task:
$full_reference

Use it as an example of the expected solution style, then produce your own complete response.
""",
    """
$question

Example response for this same task:
$full_reference

Use it as a model of the expected level of detail, structure, and task alignment.
Now produce your own answer to the original question.
""",
    """
$question

A reference-style answer to this question is shown below:
$full_reference

Treat it as an example response, then answer the question yourself.
Preserve the format requested by the original task.
""",
    """
$question

Here is an example solution for this instance:
$full_reference

Use the example to understand what a complete response should look like.
Then provide your own response to the original task.
""",
    """
$question

Example answer:
$full_reference

Use this as a sample of a good response for the task.
Now answer the original question with your own complete response.
""",
    """
$question

The following illustrates a suitable response to this task:
$full_reference

Use it as an example, not as text to mechanically repeat.
Answer the original question in the requested format.
""",
    """
$question

Reference example for this instance:
$full_reference

Let this example guide the style, structure, and level of specificity.
Now produce your own answer to the original task.
""",
    """
$question

Below is a sample response for this question:
$full_reference

Use the sample to infer what a complete answer should look like.
Then answer the original question yourself.
""",
    """
$question

A completed example response is provided here:
$full_reference

Use it as an example of the expected answer style.
Now generate your own response while following the original instructions.
""",
]


GOLD_RESPONSE_INV_PROMPT_TEMPLATES = [
    """
Here is a verified reference solution for the problem below:
$full_reference

Now solve the problem entirely by yourself:

$question
""",
]


GOLD_RESPONSE_AS_GUIDANCE_PROMPT_TEMPLATES = [
    """
$question

Helpful guidance for solving this instance:
$full_reference

Use the guidance only to improve your answer. Do not simply copy it blindly.
Now answer the original question in the required format.
""",
    """
$question

Instance-specific reference guidance:
$full_reference

Use this information to improve correctness and formatting, but write your own answer.
Answer the original question in the required format.
""",
    """
$question

Reference guidance for this instance:
$full_reference

Use this guidance to improve correctness and avoid missing key details.
Answer the original question in the required format.
""",
    """
$question

Useful information for solving this task:
$full_reference

Use the information as guidance while forming your own answer.
Follow the original task instructions.
""",
    """
$question

Instance-specific guidance:
$full_reference

Rely on this guidance to reach a better answer, but do not treat it as a template to copy.
Respond to the original question in the expected format.
""",
    """
$question

Guidance derived from the reference response:
$full_reference

Use it to check the important reasoning, constraints, and final decision.
Now answer the original task as requested.
""",
    """
$question

Helpful reference information:
$full_reference

Use this information to support your solution and avoid unnecessary deviations.
Return the answer using the format required by the task.
""",
    """
$question

Additional guidance for this exact instance:
$full_reference

Use the guidance to improve the answer's accuracy and alignment with the task.
Now solve the original question.
""",
    """
$question

Reference-based guidance:
$full_reference

Use it as support for solving the task, while still producing your own answer.
Respect all original formatting requirements.
""",
    """
$question

Guidance to consider before answering:
$full_reference

Use this guidance to identify the key details needed for the correct response.
Answer the original question in the required format.
""",
]


GOLD_HINT_V2_PROMPT_TEMPLATES = [
    """
$question

Concise oracle hint for this instance:
$short_hint

Use this hint to choose the correct actions and action inputs.
Do not copy a reference answer format.
You must reproduce every Step 1, Step 2, Step 3... from the hint.
Do not stop after the first action.
Now answer the original question in the required format.
""",
]


GOLD_HINT_V3_PROMPT_TEMPLATES = [
    """
$question

Concise oracle hint for this instance:
$short_hint

Use the hint as the complete required tool-call trace.

Output only the tool-call trace, using this format for each step:
Thought: ...
Action: ...
Action Input: ...

Rules:
- Output exactly one Action block for each listed step.
- Follow the steps in the same order.
- Do not skip, merge, repeat, or add steps.
- Use exactly the listed action names.
- Use exactly the listed Action Input fields and values.
- Do not write a final natural-language answer after the tool calls.
- Even if a later step seems to depend on a previous tool result, still output it exactly as listed.
""",
]


# =============================================================================
# Two-stage strategies: final teacher prompt templates
# =============================================================================

SELF_FEEDBACK_PROMPT_TEMPLATES = [
    """
$question

Helpful feedback for this specific instance:
$feedback_text

Now answer the original question, applying the feedback and respecting the required format.
""",
    """
$question

You previously received the following instance-specific feedback:
$feedback_text

Use this feedback to correct or improve the answer.
Now answer the original question in the required format.
""",
    """
$question

Instance-specific correction signal:
$feedback_text

Take this correction into account while solving the original task.
Return the answer using the format requested by the task.
""",
    """
$question

Additional guidance for this instance:
$feedback_text

Use the guidance to avoid the earlier issue and produce a corrected answer.
Follow the original task instructions and output format.
""",
    """
$question

Relevant note for improving the answer:
$feedback_text

Use the note as supporting guidance, then answer the original question directly.
Keep the response aligned with the required format.
""",
    """
$question

For this instance, the following feedback may help:
$feedback_text

Revise the solution accordingly and answer the original task.
Respect all formatting constraints from the original prompt.
""",
    """
$question

Correction context:
$feedback_text

Use this context to guide the next answer, especially where the previous response may have gone wrong.
Answer the original question in the expected format.
""",
    """
$question

Targeted improvement note:
$feedback_text

Apply the note while solving the task again.
Do not change the requested answer structure.
""",
    """
$question

Feedback to consider for this specific task:
$feedback_text

Use it to make the answer more accurate and better aligned with the task.
Now provide the answer in the required format.
""",
    """
$question

Instance-level advice:
$feedback_text

Let this advice guide the correction, but preserve the original task's instructions.
Produce the final response in the requested format.
""",
]


STRUCTURED_SELF_FEEDBACK_PROMPT_TEMPLATES = [
    """
$question

Concrete correction note for this instance:
$feedback_text

Apply the correction note and answer the original question in the required format.
""",
]


RATIONALIZATION_PROMPT_TEMPLATES = [
    """
$question

Useful rationale for this instance:
$feedback_text

Use the rationale to solve the original task.
Do not assume the rationale is a final answer; produce the answer in the required format.
""",
]


# =============================================================================
# Two-stage strategies: stage1 feedback/rationale templates
# =============================================================================

SELF_FEEDBACK_STAGE1_TEMPLATES = [
    """
You are creating feedback for improving a model's answer.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write only 1 or 2 concise hints, guidelines, or corrections that would help the model answer correctly.
Do not solve the full problem again.
Do not copy the reference answer verbatim.
Focus on what to do or what to avoid.
Do not start with "Thought:", "Reasoning:", "Hint:", "<reasoning>", or "<answer>".
Write the feedback as plain notes only.
""",
    """
You are writing a brief improvement note for a model response.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write 1 or 2 concise notes that would help the model improve its answer.
Focus on the most important correction, missing constraint, or wrong assumption.
Do not provide the full solution.
Do not copy the reference answer.
Avoid labels such as "Thought", "Reasoning", "Hint", "<reasoning>", or "<answer>".
Write plain feedback only.
""",
    """
Compare the model response with the reference answer and produce a short correction.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write at most two concise feedback notes.
Mention what should be corrected or checked, without re-solving the task.
Do not quote the reference answer verbatim.
Do not include answer-format tags or reasoning headers.
Keep the feedback as plain natural language.
""",
    """
You are preparing lightweight feedback to help improve an answer.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write 1 or 2 short corrections or guidelines.
Prefer concrete, instance-specific advice over generic advice.
Do not write the final answer.
Do not reproduce the reference answer.
Do not start with a heading or special tag.
""",
    """
Assess the response only to identify useful feedback.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write a compact note describing the main issue to fix or the key detail to preserve.
Use no more than two sentences.
Do not solve the task again.
Do not copy the reference wording.
Return only the feedback note.
""",
    """
Create concise feedback for a second attempt at the task.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write 1 or 2 useful notes that would guide a corrected answer.
Emphasize the decisive constraint, calculation, comparison, action, or format issue when relevant.
Do not provide a complete answer.
Do not include meta labels, XML-like tags, or step headers.
""",
    """
You are giving targeted feedback on a model's attempted answer.

Original task:
$question

Attempted answer:
$student_response

Reference answer:
$full_reference

Write a short plain-text correction that helps the model answer more accurately.
Focus on what changed between the attempted answer and the reference.
Do not reveal the full reference as the answer.
Do not use a prefixed label before the feedback.
""",
    """
Produce a concise diagnostic note for the model response.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Identify the most useful correction or caution for this instance.
Write 1 or 2 concise sentences.
Do not write the final response to the task.
Do not copy the reference answer.
Use plain notes without special formatting markers.
""",
    """
Write feedback that would help repair the model's answer.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Give only the minimal guidance needed for a better second answer.
Point to the relevant decision, constraint, calculation, or avoided mistake.
Do not include the final answer itself.
Do not use answer tags, reasoning tags, or named sections.
""",
    """
You are creating a small correction note, not a full solution.

Original task:
$question

Model response:
$student_response

Reference answer:
$full_reference

Write 1 or 2 short notes that help align the response with the reference.
Keep the note specific to this instance.
Avoid generic advice unless it is tied to a concrete correction.
Do not copy the reference answer or produce the final answer.
Return only plain feedback text.
""",
]


CONCRETE_SELF_FEEDBACK_STAGE1_TEMPLATES = [
    """
You are creating feedback for improving a model's answer.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 short bullet points of feedback.

The feedback must be concrete and instance-specific:
- mention the key correct decision, answer, action, or constraint;
- mention one specific mistake to avoid if the model's answer is wrong or incomplete.

Do not write the full final answer.
Do not copy the reference answer verbatim.
Do not give generic advice like "be careful" or "use the correct format" unless you also say what specifically should be corrected.
""",
    """
You are creating concrete feedback for improving a model's answer.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 short bullet points of feedback.

The feedback must be specific to this instance:
- mention the main correct decision, answer, action, calculation, or constraint;
- mention one concrete error, omission, or misleading step to avoid.

Do not write the final answer.
Do not copy the reference answer verbatim.
Do not give generic advice unless it includes the specific correction needed here.
Return only the two bullet points.
""",
    """
Compare the model's answer with the reference and write concrete corrective feedback.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 concise bullet points.

The first bullet should point to the key correct target or constraint.
The second bullet should identify a specific mistake, missing detail, or wrong assumption in the model's answer.

Do not solve the task again.
Do not quote the reference answer as the final response.
Do not use headings before the bullet points.
""",
    """
You are preparing two concrete notes for a better second attempt.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 short bullet points of feedback.

Make each point instance-specific and useful for correcting the answer.
Include the decisive detail needed for correctness and one mistake to avoid.
Do not provide the full final answer.
Do not copy the reference response.
Avoid vague advice that could apply to any task.
""",
    """
Create targeted feedback for repairing the model's response.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 bullet points.

Each bullet must be concrete and tied to this example.
Mention the key correction, decision, constraint, or evidence needed.
Mention one specific way the original answer was wrong, incomplete, or risky.

Do not write the corrected final response.
Do not reproduce the reference answer verbatim.
Return only the bullet points.
""",
    """
You are writing concrete correction feedback, not a solution.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 brief bullet points.

The feedback should help the model produce a corrected answer by identifying:
- the important correct target, constraint, or reasoning direction;
- one specific error or omission from the model's original answer.

Do not include the full answer.
Do not copy the reference text.
Do not add labels or sections before the bullets.
""",
    """
Generate concise, instance-specific feedback for the model's answer.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 short bullet points.

The bullets should be actionable and concrete:
- one should indicate what the corrected answer must account for;
- one should describe a particular mistake or gap in the original response.

Do not solve the problem in full.
Do not paste the reference answer.
Do not write generic comments without the concrete correction.
""",
    """
You are creating two feedback bullets for a corrected attempt.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 bullet points.

Focus on the most important instance-specific correction.
Mention the correct decision, answer feature, action, calculation, or constraint when helpful.
Mention one concrete mistake to avoid from the original response.

Do not write the final answer itself.
Do not copy the reference answer verbatim.
Keep the feedback short.
""",
    """
Write concrete feedback that would help the model fix its answer.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Output exactly 2 short bullet points.

The feedback should not be a full solution.
It should identify the key correction needed for this instance and a specific problem in the original answer.
Do not use special reasoning tags, answer tags, or section titles.
Do not copy the reference answer.
""",
    """
You are producing minimal corrective feedback for this example.

Original task:
$question

Model's original answer:
$student_response

Reference/golden answer:
$full_reference

Write exactly 2 concise bullet points.

Make the feedback concrete enough that it points to the correct target and the main avoided mistake.
Do not provide the complete final response.
Do not quote the reference answer as the answer.
Avoid broad advice such as checking the format unless you state the exact format-related issue.
""",
]


STRUCTURED_SELF_FEEDBACK_STAGE1_TEMPLATES = [
    """
You are creating a short correction note for a model's answer.

Original task:
$question

Model's original answer:
$student_response

Oracle information:
$short_hint

Write a concrete correction note with exactly two bullet points:
- Correct decision: mention the exact correct tool/action and the decisive input fields.
- Avoid: mention one specific mistake to avoid, based on the model's original answer.

Rules:
- Do not write the final answer.
- Do not copy the full oracle answer.
- Do not be vague.
- Do not say only "make sure", "ensure", or "be careful" unless you also state the exact action/input.
- Keep it under 80 words.
""",
]


RATIONALIZATION_STAGE1_TEMPLATES = [
    """
You are creating a rationale for a solved example.

Original task:
$question

Final answer / target decision:
$compact_answer

Write a concise rationale explaining how to reach this final answer.

Rules:
- Do not write the final answer itself as a standalone answer.
- Do not copy the dataset reference response verbatim.
- Explain the key reasoning, constraints, or decisions that justify the answer.
- Keep the rationale focused on what helps solve this specific instance.
- Do not start with "Thought:", "Reasoning:", "Hint:", "<reasoning>", or "<answer>".
- Write the rationale as plain notes only.
""",
]

# =============================================================================
# Two-stage gold-response rewrite strategy
# =============================================================================

GOLD_RESPONSE_REWRITE_STAGE1_TEMPLATES = [
    """
You are rewriting a reference response before it is used as a training-time example.

Original task:
$question

Original reference response:
$full_reference

Rewrite the reference response in your own words as a clean, complete response to the original task.

Strict requirements:
- Answer the original task, not a different or simplified task.
- Preserve the same final answer or final decision as the reference.
- Preserve the response format required by the original task.
- If the task expects a full response with reasoning and an answer, return a full response with both parts.
- If the task uses tags such as <reasoning> and <answer>, keep the same required tags.
- If the task is multiple-choice and the final answer is a letter, keep the same final answer letter.
- Fix minor clarity, formatting, or task-alignment problems when needed.
- Do not copy the reference response sentence by sentence.
- Do not add unrelated facts, assumptions, disclaimers, headings, or stylistic habits.
- Do not make the response longer than necessary.

Return only the rewritten full response.
""",
    """
Rewrite the gold/reference response into a normalized response for the original task.

Original task:
$question

Gold/reference response:
$full_reference

Your rewritten response must:
- be a complete response to the original task;
- preserve the reference's final answer;
- follow the original task's requested format exactly;
- include reasoning only when the original task expects reasoning or when the reference format requires it;
- keep required delimiters, tags, option letters, fields, or tool/action formats;
- remove unnecessary verbosity and stylistic artifacts;
- avoid copying the reference wording verbatim;
- avoid adding new information that is not supported by the task or the reference.

For multiple-choice tasks, preserve the same final option letter and keep it in the required answer location.

Return only the normalized full response.
""",
    """
Create a corrected paraphrase of the reference response.

Original task:
$question

Reference response:
$full_reference

Write a new version that could serve as a clean example of a valid response to the original task.

Rules:
- Preserve the final answer exactly in meaning.
- Preserve the required output structure of the original task.
- If the expected response contains both reasoning and a final answer, include both.
- If the expected response is concise, keep the rewrite concise.
- Do not output only the final answer unless the original task asks only for the final answer.
- Do not repeat the reference wording mechanically.
- Do not introduce a new style, extra commentary, or unnecessary explanation.
- Repair only minor formatting, clarity, or consistency issues.

Return only the corrected paraphrased response.
""",
    """
You are preparing a task-aligned rewritten reference response for self-distillation.

Original task:
$question

Raw reference response:
$full_reference

Produce a rewritten full response that:
- satisfies the original task requirements;
- keeps the same final answer as the raw reference;
- keeps any required tags or answer delimiters;
- keeps the same response type: full response if the task asks for a full response, final answer only if the task asks for final answer only;
- removes irrelevant verbosity, copied phrasing, and stylistic artifacts;
- keeps reasoning concise and directly tied to the final answer;
- does not add unsupported claims or alternative answers.

Return only the rewritten response.
""",
    """
Rewrite the gold response as a clean response to the original question.

Task:
$question

Gold response:
$full_reference

Checklist:
- Same final answer as the gold response.
- Same required format as the original task.
- Full response when the original task expects a full response.
- Required tags preserved when present.
- No answer-only shortcut unless the task explicitly asks for answer only.
- No verbatim copying of the gold response.
- No extra headers, apologies, meta-commentary, or stylistic additions.
- Concise, task-focused reasoning when reasoning is expected.

Return only the rewritten full response.
""",
]


GOLD_RESPONSE_REWRITE_AS_EXAMPLE_PROMPT_TEMPLATES = [
    """
$question

Below is a rewritten reference response for this same task:
$feedback_text

Use it only as a reference for correctness, format, and level of detail.

Now solve the original task yourself.
Your response must satisfy the original task requirements.
If the original task expects a full response, provide a full response, not only the final answer.
Do not copy the reference response verbatim.
""",
    """
$question

A cleaned reference response for this task is provided below:
$feedback_text

Use it as an example of a valid task-aligned response, not as text to repeat.

Answer the original task in your own words.
Preserve the required output format.
Include reasoning if the original task expects reasoning.
Do not output only the final answer unless the original task explicitly asks for only the final answer.
""",
    """
$question

Rewritten reference example:
$feedback_text

Use the example to understand the expected response structure and final-answer format.

Now answer the original question yourself.
Follow the original instructions exactly.
Do not mechanically copy the example.
Do not add unnecessary stylistic phrases or extra commentary.
""",
    """
$question

Here is a normalized example response for the same task:
$feedback_text

Treat it as a reference example for format and correctness.

Produce your own complete response to the original task.
Keep any required tags, fields, option letters, or answer delimiters.
Avoid both answer-only shortcuts and verbatim repetition of the example.
""",
    """
$question

Clean reference-style response:
$feedback_text

Use this as a guide for what a valid response should contain.

Now solve the original task again.
Match the original task requirements.
If the task expects <reasoning> and <answer>, include both tags.
If the task is multiple-choice, put the final option in the required answer location.
Do not copy the example sentence by sentence.
""",
    """
$question

A task-aligned rewritten reference response is shown below:
$feedback_text

Use it to infer the required format and appropriate concision.

Now provide your own response to the original task.
Your response should be complete for the task, but no longer than necessary.
Do not imitate irrelevant wording, tone, or phrasing from the example.
""",
    """
$question

Reference response after cleanup:
$feedback_text

Use this example for correctness and formatting only.

Answer the original task yourself.
Preserve the expected response structure.
Do not reduce the response to only the final answer unless that is what the original task requests.
Do not repeat the example verbatim.
""",
    """
$question

Corrected paraphrase of the reference response:
$feedback_text

Use the corrected paraphrase as a non-verbatim example.

Now solve the original task.
Follow the original task instructions, including any required reasoning, tags, answer fields, or option-letter format.
Avoid adding a new style or extra explanatory habits.
""",
    """
$question

Cleaned example response:
$feedback_text

This example shows the expected format and level of detail.

Write your own response to the original task.
Keep the same required answer structure.
Do not simply restate or copy the example.
Do not omit required reasoning or required answer tags.
""",
    """
$question

Normalized reference example:
$feedback_text

Use it as a reference for the expected answer format and final decision.

Now answer the original question in your own words.
Return a complete response according to the original instructions.
Avoid answer-only shortcuts, verbatim copying, and unnecessary stylistic additions.
""",
]


# =============================================================================
# Two-stage gold-response rewrite strategy v2
# =============================================================================

GOLD_RESPONSE_REWRITE_V2_STAGE1_TEMPLATES = [
    """
You are rewriting a reference response before it is used as a training-time example.

Original task:
$question

Original reference response:
$full_reference

Rewrite the reference response in your own words as a clean, complete response to the original task.

Strict requirements:
- Answer the original task, not a different or simplified task.
- Preserve the same final answer or final decision as the reference.
- Preserve the response format required by the original task.
- If the reference response contains both reasoning and a final answer, return a full response with both parts.
- If the reference response uses tags such as <reasoning> and <answer>, keep the same required tags.
- If the task is multiple-choice and the final answer is a letter, keep the same final answer letter.
- Fix minor clarity, formatting, or task-alignment problems when needed.
- Do not copy the reference response sentence by sentence.
- Do not add unrelated facts, assumptions, disclaimers, headings, or stylistic habits.
- Do not make the response longer than necessary.

Return only the rewritten full response.
""",
#     """
# Rewrite the gold/reference response into a normalized response for the original task.

# Original task:
# $question

# Gold/reference response:
# $full_reference

# Your rewritten response must:
# - be a complete response to the original task;
# - preserve the reference's final answer;
# - follow the original task's requested format exactly;
# - preserve reasoning whenever the reference response contains reasoning;
# - keep required delimiters, tags, option letters, fields, or tool/action formats;
# - never collapse a reference response with reasoning into only the final answer;
# - remove unnecessary verbosity and stylistic artifacts;
# - avoid copying the reference wording verbatim;
# - avoid adding new information that is not supported by the task or the reference.

# For multiple-choice tasks, preserve the same final option letter and keep it in the required answer location.

# Return only the normalized full response.
# """,
    """
Rewrite the gold/reference response into a structurally faithful paraphrase.

Original task:
$question

Gold/reference response:
$full_reference

Your rewritten response must preserve the structure of the gold/reference response.

Structural requirements:
- If the reference response has multiple parts, your rewrite must have the same parts in the same order.
- If the reference response has reasoning plus a final answer, your rewrite must have reasoning plus a final answer.
- If the reference response uses tags, delimiters, fields, or sections, your rewrite must preserve them.
- Do not replace a full reference response with only its final-answer part.
- Do not infer a shorter response format from the original task when the reference response is full.

Content requirements:
- Preserve the reference's final answer.
- For multiple-choice tasks, preserve the same final option letter in the final-answer location.
- Rewrite the reasoning in your own words without changing the conclusion.
- Remove only unnecessary verbosity or stylistic artifacts.
- Do not copy the reference wording verbatim.
- Do not add new information that is not supported by the task or the reference.

Return only the rewritten full response.
""",
    """
Create a corrected paraphrase of the reference response.

Original task:
$question

Reference response:
$full_reference

Write a new version that could serve as a clean example of a valid response to the original task.

Rules:
- Preserve the final answer exactly in meaning.
- Preserve the required output structure of the original task.
- If the reference response contains both reasoning and a final answer, include both.
- If the expected response is concise, keep the rewrite concise without dropping required reasoning or tags.
- Do not output only the final answer unless the reference response itself contains only the final answer.
- Do not repeat the reference wording mechanically.
- Do not introduce a new style, extra commentary, or unnecessary explanation.
- Repair only minor formatting, clarity, or consistency issues.

Return only the corrected paraphrased response.
""",
    """
You are preparing a task-aligned rewritten reference response for self-distillation.

Original task:
$question

Raw reference response:
$full_reference

Produce a rewritten full response that:
- satisfies the original task requirements;
- keeps the same final answer as the raw reference;
- keeps any required tags or answer delimiters;
- keeps the same response type: full response if the reference response is a full response, final answer only if the reference response is final-answer only;
- preserves reasoning whenever the raw reference response contains reasoning;
- removes irrelevant verbosity, copied phrasing, and stylistic artifacts;
- keeps reasoning concise and directly tied to the final answer;
- does not add unsupported claims or alternative answers.

Return only the rewritten response.
""",
#     """
# Rewrite the gold response as a clean response to the original question.

# Task:
# $question

# Gold response:
# $full_reference

# Checklist:
# - Same final answer as the gold response.
# - Same required format as the original task.
# - Full response when the gold response is a full response.
# - Required tags preserved when present.
# - No answer-only shortcut when the gold response contains reasoning or required response tags.
# - No verbatim copying of the gold response.
# - No extra headers, apologies, meta-commentary, or stylistic additions.
# - Concise, task-focused reasoning when reasoning is present in the gold response.

# Return only the rewritten full response.
# """,
    """
Rewrite the gold response as a structurally faithful response to the original question.

Task:
$question

Gold response:
$full_reference

Use the gold response as the structural template for your rewrite.

Checklist:
- Preserve the same final answer as the gold response.
- Preserve the same response structure as the gold response.
- If the gold response has multiple parts, keep all of those parts.
- If the gold response contains reasoning before the final answer, include reasoning before the final answer.
- If the gold response uses tags, delimiters, fields, or answer sections, preserve them.
- Do not shorten a full gold response into only the final-answer part.
- Do not treat the final-answer part as the whole response unless the gold response itself is final-answer only.
- Rewrite in your own words rather than copying sentence by sentence.
- Do not add extra headers, apologies, meta-commentary, unsupported facts, or stylistic additions.
- Keep the rewritten reasoning concise, but do not omit required reasoning.

Return only the rewritten full response.
""",
]


GOLD_RESPONSE_REWRITE_AS_EXAMPLE_V2_PROMPT_TEMPLATES = [
    """
$question

Below is a rewritten reference response for this same task:
$feedback_text

Use it only as a reference for correctness, format, and level of detail.

Now solve the original task yourself.
Your response must satisfy the original task requirements.
If the original task expects a full response, provide a full response, not only the final answer.
Do not copy the reference response verbatim.
""",
    """
$question

A cleaned reference response for this task is provided below:
$feedback_text

Use it as an example of a valid task-aligned response, not as text to repeat.

Answer the original task in your own words.
Preserve the required output format.
Include reasoning if the reference example contains reasoning.
Do not output only the final answer unless the reference example contains only the final answer.
""",
    """
$question

Rewritten reference example:
$feedback_text

Use the example to understand the expected response structure and final-answer format.

Now answer the original question yourself.
Follow the original instructions exactly.
If the example contains reasoning and an answer section, preserve that response structure.
Do not mechanically copy the example.
Do not add unnecessary stylistic phrases or extra commentary.
""",
    """
$question

Here is a normalized example response for the same task:
$feedback_text

Treat it as a reference example for format and correctness.

Produce your own complete response to the original task.
Keep any required tags, fields, option letters, or answer delimiters.
Avoid both answer-only shortcuts and verbatim repetition of the example.
""",
    """
$question

Clean reference-style response:
$feedback_text

Use this as a guide for what a valid response should contain.

Now solve the original task again.
Match the original task requirements.
If the task expects <reasoning> and <answer>, include both tags.
If the task is multiple-choice, put the final option in the required answer location.
Do not copy the example sentence by sentence.
""",
    """
$question

A task-aligned rewritten reference response is shown below:
$feedback_text

Use it to infer the required format and appropriate level of detail.

Now provide your own response to the original task.
Your response should be complete for the task, with concise reasoning when the example contains reasoning.
If the example contains both reasoning and a final answer, do not reduce your response to only the final answer.
Do not imitate irrelevant wording, tone, or phrasing from the example.
""",
    """
$question

Reference response after cleanup:
$feedback_text

Use this example for correctness and formatting only.

Answer the original task yourself.
Preserve the expected response structure.
Do not reduce the response to only the final answer unless that is what the original task requests.
Do not repeat the example verbatim.
""",
    """
$question

Corrected paraphrase of the reference response:
$feedback_text

Use the corrected paraphrase as a non-verbatim example.

Now solve the original task.
Follow the response structure shown in the example, including required reasoning, tags, answer fields, or option-letter format.
If the example contains reasoning, include concise reasoning in your own response.
Do not reduce the response to only the final answer.
Avoid adding a new style or extra explanatory habits.
""",
    """
$question

Cleaned example response:
$feedback_text

This example shows the expected format and level of detail.

Write your own response to the original task.
Keep the same required answer structure.
Do not simply restate or copy the example.
Do not omit required reasoning or required answer tags.
""",
    """
$question

Normalized reference example:
$feedback_text

Use it as a reference for the expected answer format and final decision.

Now answer the original question in your own words.
Return a complete response according to the original instructions.
Avoid answer-only shortcuts, verbatim copying, and unnecessary stylistic additions.
""",
]


# =============================================================================
# SDPO-like sibling/feedback self-teacher strategy
# =============================================================================

SDPO_SIBLING_FEEDBACK_PROMPT_TEMPLATES = [
    """
$question

${sdpo_solution_block}${sdpo_feedback_block}Correctly solve the original question.
""",
    """
$question

${sdpo_solution_block}${sdpo_feedback_block}Now provide a correct response to the original task.
Follow the original task requirements.
""",
    """
$question

${sdpo_solution_block}${sdpo_feedback_block}Use the information above only to correct the original task response.
Now solve the original task correctly.
""",
]
