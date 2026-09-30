


#!/usr/bin/env python3
from __future__ import annotations

"""Core definitions for the Spatial Contradiction benchmark.

The benchmark keeps a fixed coordinate readout while changing the latent meaning
of spatial-relation words.  In the default world, relation words use ordinary
2D grid semantics.  In the contradiction world, every direction vector is
rotated 90 degrees clockwise (R90), while the coordinate tuple itself is read in
the same frame.  The transformation is deliberately *not* stated in the model
prompt.

This file has no heavyweight dependencies.  Dataset construction, scoring, and
unit tests can therefore import it without requiring Hugging Face Datasets,
Transformers, vLLM, or an API client.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import random
import re
from typing import Any, Iterable, Mapping, Sequence

VERSION = "1.2.4"
TRANSFORMATION_VERSION = 1
TRANSFORMATION_NAME = "direction_semantics_r90_clockwise_fixed_coordinate_frame"
ROTATING_TRANSFORMATION_NAME = "direction_semantics_r90_clockwise_per_hop_rotating_frame"

OPENR1_SPATIAL_PROMPT_TEMPLATE = (
    "You will be given a spatial reasoning problem on a square grid.\n"
    "Distances are measured in grid cells. A k-cell diagonal relation means "
    "k diagonal grid steps.\n"
    "Please reason step by step, and put your final coordinate pair within "
    "\\boxed{{(x, y)}}:\n"
    "{problem}"
)

DATASET_DEFAULT_TEACHER_TEMPLATE = (
    "{question}\n\n"
    "Here is a verified reference solution for the same problem:\n"
    "{reference_response}\n\n"
    "Now solve the original problem yourself. Keep the requested final-answer format."
)


@dataclass(frozen=True)
class PhraseSpec:
    """One unambiguous surface realization of a spatial direction.

    ``semantic_family`` records the lexical/semantic cluster used to express the
    same underlying vector. ``grammar_roles`` explicitly states the syntactic
    slots in which the phrase is natural. This prevents semantically valid but
    awkward combinations such as ``two cells northward`` after a copular frame.
    """

    text: str
    semantic_family: str
    grammar_roles: tuple[str, ...]


@dataclass(frozen=True)
class DirectionSpec:
    key: str
    ordinary_vector: tuple[int, int]
    opposite: str
    relative_phrases: tuple[PhraseSpec, ...]
    motion_phrases: tuple[PhraseSpec, ...]
    canonical_phrase: str


@dataclass(frozen=True)
class StatementFamily:
    """A grammatical question-side realization for a direction phrase."""

    key: str
    phrase_kind: str
    phrase_role: str
    syntax_family: str
    template: str
    weight: float = 1.0


@dataclass(frozen=True)
class SolutionStepFamily:
    """A grammar-safe SFT-target realization for one coordinate update."""

    key: str
    phrase_role: str
    syntax_family: str
    template: str
    weight: float = 1.0


@dataclass(frozen=True)
class RenderedStatement:
    """Fully rendered statement plus provenance needed for analysis."""

    text: str
    family_id: int
    family_key: str
    syntax_family: str
    phrase_id: int
    phrase_kind: str
    phrase_semantic_family: str
    phrase_grammar_role: str
    amount_text: str


# Grammar roles are intentionally semantic rather than template-specific. A
# phrase may belong to several roles when it reads naturally in each one.
RELATIVE_PREDICATE = "relative_predicate"
RELATIVE_FRONTED = "relative_fronted"
MOTION_PREDICATIVE = "motion_predicative"
MOTION_ACTION = "motion_action"
TURN_ACTION = "turn_action"
PHRASE_GRAMMAR_ROLES: tuple[str, ...] = (
    RELATIVE_PREDICATE,
    RELATIVE_FRONTED,
    MOTION_PREDICATIVE,
    MOTION_ACTION,
    TURN_ACTION,
)


def _phrases(
    semantic_family: str,
    grammar_roles: tuple[str, ...],
    *texts: str,
) -> tuple[PhraseSpec, ...]:
    """Compact helper used only while defining the immutable vocabulary."""

    return tuple(
        PhraseSpec(text=value, semantic_family=semantic_family, grammar_roles=grammar_roles)
        for value in texts
    )


def _rel(*texts: str, family: str, fronted: bool = True) -> tuple[PhraseSpec, ...]:
    roles = (RELATIVE_PREDICATE, RELATIVE_FRONTED) if fronted else (RELATIVE_PREDICATE,)
    return _phrases(family, roles, *texts)


def _motion(
    *texts: str,
    family: str,
    predicative: bool = False,
) -> tuple[PhraseSpec, ...]:
    roles = (MOTION_ACTION, MOTION_PREDICATIVE) if predicative else (MOTION_ACTION,)
    return _phrases(family, roles, *texts)


CLOCK_LABELS: dict[str, str] = {
    "N": "12 o'clock",
    "NE": "1:30",
    "E": "3 o'clock",
    "SE": "4:30",
    "S": "6 o'clock",
    "SW": "7:30",
    "W": "9 o'clock",
    "NW": "10:30",
}

COMPASS_BEARINGS: dict[str, int] = {
    "N": 0,
    "NE": 45,
    "E": 90,
    "SE": 135,
    "S": 180,
    "SW": 225,
    "W": 270,
    "NW": 315,
}

CLOCKWISE_DIRECTION_ORDER: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
LONG_DIRECTION_NAMES: dict[str, str] = {
    "N": "north",
    "NE": "northeast",
    "E": "east",
    "SE": "southeast",
    "S": "south",
    "SW": "southwest",
    "W": "west",
    "NW": "northwest",
}


def _clock_phrases(direction: str) -> tuple[tuple[PhraseSpec, ...], tuple[PhraseSpec, ...]]:
    label = CLOCK_LABELS[direction]
    relative = _rel(
        f"in the {label} direction from",
        f"along the {label} direction from",
        family="clock_face",
    )
    motion = (
        *_motion(f"toward {label}", family="clock_face"),
        *_motion(
            f"in the {label} direction",
            f"along the {label} direction",
            f"toward the {label} position on a clock face",
            family="clock_face",
            predicative=True,
        ),
    )
    return relative, motion


def _bearing_phrases(direction: str) -> tuple[tuple[PhraseSpec, ...], tuple[PhraseSpec, ...]]:
    degrees = COMPASS_BEARINGS[direction]
    relative = _rel(
        f"on a compass bearing of {degrees} degrees from",
        f"along a {degrees}-degree compass bearing from",
        family="compass_bearing",
    )
    motion = _motion(
        f"along a compass bearing of {degrees} degrees",
        f"on a {degrees}-degree compass bearing",
        f"along a {degrees}-degree compass heading",
        family="compass_bearing",
        predicative=True,
    )
    return relative, motion


def _oriented_turn_phrases(direction: str) -> tuple[PhraseSpec, ...]:
    """Generate compositional egocentric instructions resolving to ``direction``.

    A right/clockwise quarter-turn starts two 45-degree bins counterclockwise
    from the target; a left/counterclockwise quarter-turn starts two bins
    clockwise from it. The opposite heading supplies a 180-degree variant.
    These constructions remain compositionally consistent under the benchmark's
    R90 semantics: rotating the named initial heading and preserving the
    egocentric turn rotates the resulting travel direction by the same amount.
    """

    index = CLOCKWISE_DIRECTION_ORDER.index(direction)
    right_start = CLOCKWISE_DIRECTION_ORDER[(index - 2) % len(CLOCKWISE_DIRECTION_ORDER)]
    left_start = CLOCKWISE_DIRECTION_ORDER[(index + 2) % len(CLOCKWISE_DIRECTION_ORDER)]
    opposite_start = CLOCKWISE_DIRECTION_ORDER[(index + 4) % len(CLOCKWISE_DIRECTION_ORDER)]
    right_name = LONG_DIRECTION_NAMES[right_start]
    left_name = LONG_DIRECTION_NAMES[left_start]
    opposite_name = LONG_DIRECTION_NAMES[opposite_start]
    texts = (
        f"face {right_name}, make a quarter-turn clockwise, then move forward",
        f"face {right_name}, turn right by 90 degrees, then move forward",
        f"face {left_name}, make a quarter-turn counterclockwise, then move forward",
        f"face {left_name}, turn left by 90 degrees, then move forward",
        f"face {opposite_name}, turn around, then move forward",
    )
    return _phrases("oriented_turn", (TURN_ACTION,), *texts)


# Vocabulary policy
# -----------------
# Every phrase denotes exactly one of the eight grid directions in ordinary
# English. The entire relation inventory rotates together in the R90 world.
# Richness comes from lexical families *and* grammar-safe surface roles. Clock
# and bearing formulations are treated as alternative surface descriptions of
# the same eight directions; oriented-turn instructions compose an absolute
# heading with a deterministic egocentric turn.

def _direction_spec(
    key: str,
    ordinary_vector: tuple[int, int],
    opposite: str,
    relative_core: tuple[PhraseSpec, ...],
    motion_core: tuple[PhraseSpec, ...],
    canonical_phrase: str,
) -> DirectionSpec:
    clock_rel, clock_motion = _clock_phrases(key)
    bearing_rel, bearing_motion = _bearing_phrases(key)
    return DirectionSpec(
        key=key,
        ordinary_vector=ordinary_vector,
        opposite=opposite,
        relative_phrases=relative_core + clock_rel + bearing_rel,
        motion_phrases=motion_core + clock_motion + bearing_motion + _oriented_turn_phrases(key),
        canonical_phrase=canonical_phrase,
    )


DIRECTION_SPECS: dict[str, DirectionSpec] = {
    "N": _direction_spec(
        "N", (0, 1), "S",
        (
            *_rel("north of", "to the north of", "directly north of", "due north of", "straight north of", family="compass"),
            *_rel("above", "directly above", "straight above", family="screen_relative"),
            *_rel("up from", "upward from", family="screen_relative", fronted=False),
            *_rel("vertically above", family="axis_reinforced"),
            *_rel("on the same vertical line above", family="axis_reinforced"),
            *_rel("northward from", family="vector_motion", fronted=False),
        ),
        (
            *_motion("north", "to the north", "due north", "straight north", "directly north", family="compass", predicative=True),
            *_motion("toward the north", family="compass"),
            *_motion("up", "straight up", "upward", "straight upward", family="screen_relative"),
            *_motion("vertically up", "vertically upward", family="axis_reinforced"),
            *_motion("northward", family="vector_motion"),
        ),
        "north of",
    ),
    "NE": _direction_spec(
        "NE", (1, 1), "SW",
        (
            *_rel("northeast of", "to the northeast of", "directly northeast of", family="compass"),
            *_rel("to the upper right of", "to the top right of", family="screen_relative"),
            *_rel("up-right of", family="screen_relative", fronted=False),
            *_rel("diagonally above and to the right of", "diagonally up and to the right of", "diagonally to the upper right of", "diagonally to the top right of", family="compositional"),
            *_rel("northeastward from", "diagonally northeast from", family="vector_motion", fronted=False),
            *_rel("along the northeast diagonal from", "along the upper-right diagonal from", family="diagonal_path"),
        ),
        (
            *_motion("northeast", "to the northeast", family="compass", predicative=True),
            *_motion("toward the northeast", family="compass"),
            *_motion("toward the upper right", "toward the top right", family="screen_relative", predicative=True),
            *_motion("up-right", family="screen_relative"),
            *_motion("diagonally up and to the right", "diagonally toward the upper right", "diagonally toward the top right", family="compositional", predicative=True),
            *_motion("northeastward", "diagonally northeast", family="vector_motion"),
            *_motion("along the northeast diagonal", "along the upper-right diagonal", family="diagonal_path", predicative=True),
        ),
        "northeast of",
    ),
    "E": _direction_spec(
        "E", (1, 0), "W",
        (
            *_rel("east of", "to the east of", "directly east of", "due east of", "straight east of", family="compass"),
            *_rel("to the right of", "directly to the right of", "straight to the right of", "right of", family="screen_relative"),
            *_rel("horizontally to the right of", family="axis_reinforced"),
            *_rel("eastward from", family="vector_motion", fronted=False),
        ),
        (
            *_motion("east", "to the east", "due east", "straight east", "directly east", family="compass", predicative=True),
            *_motion("toward the east", family="compass"),
            *_motion("to the right", "straight to the right", family="screen_relative", predicative=True),
            *_motion("right", "rightward", family="screen_relative"),
            *_motion("horizontally to the right", family="axis_reinforced", predicative=True),
            *_motion("eastward", family="vector_motion"),
        ),
        "east of",
    ),
    "SE": _direction_spec(
        "SE", (1, -1), "NW",
        (
            *_rel("southeast of", "to the southeast of", "directly southeast of", family="compass"),
            *_rel("to the lower right of", "to the bottom right of", family="screen_relative"),
            *_rel("down-right of", family="screen_relative", fronted=False),
            *_rel("diagonally below and to the right of", "diagonally down and to the right of", "diagonally to the lower right of", "diagonally to the bottom right of", family="compositional"),
            *_rel("southeastward from", "diagonally southeast from", family="vector_motion", fronted=False),
            *_rel("along the southeast diagonal from", "along the lower-right diagonal from", family="diagonal_path"),
        ),
        (
            *_motion("southeast", "to the southeast", family="compass", predicative=True),
            *_motion("toward the southeast", family="compass"),
            *_motion("toward the lower right", "toward the bottom right", family="screen_relative", predicative=True),
            *_motion("down-right", family="screen_relative"),
            *_motion("diagonally down and to the right", "diagonally toward the lower right", "diagonally toward the bottom right", family="compositional", predicative=True),
            *_motion("southeastward", "diagonally southeast", family="vector_motion"),
            *_motion("along the southeast diagonal", "along the lower-right diagonal", family="diagonal_path", predicative=True),
        ),
        "southeast of",
    ),
    "S": _direction_spec(
        "S", (0, -1), "N",
        (
            *_rel("south of", "to the south of", "directly south of", "due south of", "straight south of", family="compass"),
            *_rel("below", "directly below", "straight below", family="screen_relative"),
            *_rel("down from", "downward from", family="screen_relative", fronted=False),
            *_rel("vertically below", family="axis_reinforced"),
            *_rel("on the same vertical line below", family="axis_reinforced"),
            *_rel("southward from", family="vector_motion", fronted=False),
        ),
        (
            *_motion("south", "to the south", "due south", "straight south", "directly south", family="compass", predicative=True),
            *_motion("toward the south", family="compass"),
            *_motion("down", "straight down", "downward", "straight downward", family="screen_relative"),
            *_motion("vertically down", "vertically downward", family="axis_reinforced"),
            *_motion("southward", family="vector_motion"),
        ),
        "south of",
    ),
    "SW": _direction_spec(
        "SW", (-1, -1), "NE",
        (
            *_rel("southwest of", "to the southwest of", "directly southwest of", family="compass"),
            *_rel("to the lower left of", "to the bottom left of", family="screen_relative"),
            *_rel("down-left of", family="screen_relative", fronted=False),
            *_rel("diagonally below and to the left of", "diagonally down and to the left of", "diagonally to the lower left of", "diagonally to the bottom left of", family="compositional"),
            *_rel("southwestward from", "diagonally southwest from", family="vector_motion", fronted=False),
            *_rel("along the southwest diagonal from", "along the lower-left diagonal from", family="diagonal_path"),
        ),
        (
            *_motion("southwest", "to the southwest", family="compass", predicative=True),
            *_motion("toward the southwest", family="compass"),
            *_motion("toward the lower left", "toward the bottom left", family="screen_relative", predicative=True),
            *_motion("down-left", family="screen_relative"),
            *_motion("diagonally down and to the left", "diagonally toward the lower left", "diagonally toward the bottom left", family="compositional", predicative=True),
            *_motion("southwestward", "diagonally southwest", family="vector_motion"),
            *_motion("along the southwest diagonal", "along the lower-left diagonal", family="diagonal_path", predicative=True),
        ),
        "southwest of",
    ),
    "W": _direction_spec(
        "W", (-1, 0), "E",
        (
            *_rel("west of", "to the west of", "directly west of", "due west of", "straight west of", family="compass"),
            *_rel("to the left of", "directly to the left of", "straight to the left of", "left of", family="screen_relative"),
            *_rel("horizontally to the left of", family="axis_reinforced"),
            *_rel("westward from", family="vector_motion", fronted=False),
        ),
        (
            *_motion("west", "to the west", "due west", "straight west", "directly west", family="compass", predicative=True),
            *_motion("toward the west", family="compass"),
            *_motion("to the left", "straight to the left", family="screen_relative", predicative=True),
            *_motion("left", "leftward", family="screen_relative"),
            *_motion("horizontally to the left", family="axis_reinforced", predicative=True),
            *_motion("westward", family="vector_motion"),
        ),
        "west of",
    ),
    "NW": _direction_spec(
        "NW", (-1, 1), "SE",
        (
            *_rel("northwest of", "to the northwest of", "directly northwest of", family="compass"),
            *_rel("to the upper left of", "to the top left of", family="screen_relative"),
            *_rel("up-left of", family="screen_relative", fronted=False),
            *_rel("diagonally above and to the left of", "diagonally up and to the left of", "diagonally to the upper left of", "diagonally to the top left of", family="compositional"),
            *_rel("northwestward from", "diagonally northwest from", family="vector_motion", fronted=False),
            *_rel("along the northwest diagonal from", "along the upper-left diagonal from", family="diagonal_path"),
        ),
        (
            *_motion("northwest", "to the northwest", family="compass", predicative=True),
            *_motion("toward the northwest", family="compass"),
            *_motion("toward the upper left", "toward the top left", family="screen_relative", predicative=True),
            *_motion("up-left", family="screen_relative"),
            *_motion("diagonally up and to the left", "diagonally toward the upper left", "diagonally toward the top left", family="compositional", predicative=True),
            *_motion("northwestward", "diagonally northwest", family="vector_motion"),
            *_motion("along the northwest diagonal", "along the upper-left diagonal", family="diagonal_path", predicative=True),
        ),
        "northwest of",
    ),
}

PHRASE_SEMANTIC_FAMILIES: tuple[str, ...] = tuple(
    sorted(
        {
            phrase.semantic_family
            for spec in DIRECTION_SPECS.values()
            for phrase in (*spec.relative_phrases, *spec.motion_phrases)
        }
    )
)

DIRECTION_KEYS: tuple[str, ...] = tuple(DIRECTION_SPECS)
DIAGONAL_DIRECTIONS = frozenset({"NE", "SE", "SW", "NW"})

# A bounded, deliberately unambiguous entity vocabulary.  Names are shared
# across splits because the task is to generalize the relation semantics, not
# to generalize to unseen names.  Exact semantic instances and prompts remain
# disjoint across splits.
ENTITY_NAMES: tuple[str, ...] = (
    "Ari", "Bela", "Cora", "Dax", "Elin", "Farah", "Gio", "Hana",
    "Ivo", "Jade", "Kian", "Lina", "Milo", "Nia", "Omar", "Pia",
    "Quin", "Rhea", "Soren", "Tia", "Ugo", "Vera", "Wren", "Xavi",
    "Yuna", "Zed", "Alma", "Boris", "Cleo", "Devin", "Esme", "Finn",
    "Gia", "Hugo", "Iris", "Juno", "Kai", "Lora", "Mina", "Nico",
    "Opal", "Ravi", "Sara", "Theo", "Una", "Vito", "Will", "Xena",
    "Yara", "Zane", "Ayla", "Bruno", "Dina", "Evan", "Faye", "Galen",
    "Hope", "Ivan", "Kara", "Leon", "Mara", "Nora", "Orin", "Rosa",
)

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}

ANCHOR_TEMPLATES: tuple[str, ...] = (
    "{anchor} is located at {anchor_coord}.",
    "Take {anchor}'s position to be {anchor_coord}.",
    "Use {anchor} as the reference point, located at {anchor_coord}.",
    "On the grid, {anchor} is at {anchor_coord}.",
    "The coordinates of {anchor} are {anchor_coord}.",
    "Place {anchor} at {anchor_coord} on the grid.",
)

QUERY_TEMPLATES: tuple[str, ...] = (
    "What are the coordinates of {target}?",
    "Determine the coordinates of {target}.",
    "Find the coordinate pair for {target}.",
    "Starting from {anchor}'s known position, determine the coordinates of {target}.",
    "Based on these spatial relations, what coordinate pair corresponds to {target}?",
    "Where is {target} located on the grid? Give its coordinates.",
)

# Grammatical diversity is represented separately from direction-phrase
# diversity.  ``phrase_kind`` guarantees that only grammatically compatible
# phrase inventories are injected into a template.  Less common fronted
# locatives receive a lower sampling weight because they are natural but more
# marked in English.
STATEMENT_FAMILIES: tuple[StatementFamily, ...] = (
    # Relative-predicate frames.
    StatementFamily("relative_is", "relative", RELATIVE_PREDICATE, "declarative", "{subject} is {amount} {phrase} {reference}."),
    StatementFamily("relative_lies", "relative", RELATIVE_PREDICATE, "declarative", "{subject} lies {amount} {phrase} {reference}."),
    StatementFamily("relative_positioned", "relative", RELATIVE_PREDICATE, "descriptive", "{subject} is positioned {amount} {phrase} {reference}."),
    StatementFamily("relative_situated", "relative", RELATIVE_PREDICATE, "descriptive", "{subject} is situated {amount} {phrase} {reference}."),
    StatementFamily("relative_grid", "relative", RELATIVE_PREDICATE, "grid_locative", "On the grid, {subject} is {amount} {phrase} {reference}."),
    StatementFamily("relative_position_of", "relative", RELATIVE_PREDICATE, "descriptive", "The position of {subject} is {amount} {phrase} {reference}."),
    StatementFamily("relative_located", "relative", RELATIVE_PREDICATE, "descriptive", "{subject} is located {amount} {phrase} {reference}."),
    # Fronted locatives use only phrases explicitly licensed for this role.
    StatementFamily("relative_front_location", "relative", RELATIVE_FRONTED, "fronted_locative", "At the point {amount} {phrase} {reference}, {subject} is located.", weight=0.85),
    StatementFamily("relative_front_occupied", "relative", RELATIVE_FRONTED, "fronted_locative", "{subject} occupies the point {amount} {phrase} {reference}.", weight=0.90),
    StatementFamily("relative_front_where", "relative", RELATIVE_FRONTED, "fronted_locative", "The point {amount} {phrase} {reference} is where {subject} is located.", weight=0.85),
    StatementFamily("relative_look", "relative", RELATIVE_FRONTED, "discovery_locative", "Look {amount} {phrase} {reference} to find {subject}.", weight=0.75),
    # Copular motion frames are restricted to phrases that sound natural after
    # an amount (e.g. ``two cells north`` or ``two cells to the right``).
    StatementFamily("motion_from", "motion", MOTION_PREDICATIVE, "declarative", "From {reference}, {subject} is {amount} {phrase}."),
    StatementFamily("motion_relative", "motion", MOTION_PREDICATIVE, "declarative", "Relative to {reference}, {subject} is {amount} {phrase}."),
    StatementFamily("motion_viewed", "motion", MOTION_PREDICATIVE, "descriptive", "Viewed from {reference}, {subject} lies {amount} {phrase}."),
    # General movement/action frames.
    StatementFamily("motion_instruction", "motion", MOTION_ACTION, "instruction", "Starting at {reference}, move {amount} {phrase} to reach {subject}."),
    StatementFamily("motion_reached", "motion", MOTION_ACTION, "route_description", "{subject} can be reached from {reference} by moving {amount} {phrase}."),
    StatementFamily("motion_beginning", "motion", MOTION_ACTION, "instruction", "Beginning at {reference}, go {amount} {phrase}; you arrive at {subject}."),
    StatementFamily("motion_to_get", "motion", MOTION_ACTION, "route_description", "To get from {reference} to {subject}, travel {amount} {phrase}."),
    StatementFamily("motion_leads", "motion", MOTION_ACTION, "route_description", "Moving {amount} {phrase} from {reference} leads to {subject}."),
    StatementFamily("motion_conditional", "motion", MOTION_ACTION, "conditional", "If you move {amount} {phrase} from {reference}, you reach {subject}."),
    StatementFamily("motion_travel", "motion", MOTION_ACTION, "instruction", "Travel {amount} {phrase} from {reference} to arrive at {subject}."),
    StatementFamily("motion_brings", "motion", MOTION_ACTION, "route_description", "Starting from {reference}, going {amount} {phrase} brings you to {subject}."),
    StatementFamily("motion_route_goes", "motion", MOTION_ACTION, "route_description", "The route from {reference} to {subject} goes {amount} {phrase}.", weight=0.90),
    StatementFamily("motion_move_of", "motion", MOTION_ACTION, "route_description", "A move of {amount} {phrase} from {reference} ends at {subject}.", weight=0.90),
    StatementFamily("motion_follow", "motion", MOTION_ACTION, "route_description", "Follow a path {amount} {phrase} from {reference} to reach {subject}.", weight=0.90),
    # Orientation/turn compositions put the distance after ``move forward``.
    StatementFamily("turn_start", "motion", TURN_ACTION, "oriented_turn", "Starting at {reference}, {phrase} for {amount} to reach {subject}."),
    StatementFamily("turn_to_get", "motion", TURN_ACTION, "oriented_turn", "To get from {reference} to {subject}, {phrase} for {amount}."),
    StatementFamily("turn_from", "motion", TURN_ACTION, "oriented_turn", "From {reference}, {phrase} for {amount}; this brings you to {subject}."),
    StatementFamily("turn_conditional", "motion", TURN_ACTION, "oriented_turn", "You reach {subject} from {reference} if you {phrase} for {amount}."),
    StatementFamily("turn_route", "motion", TURN_ACTION, "oriented_turn", "The route from {reference} to {subject} is: {phrase} for {amount}."),
)

SOLUTION_START_TEMPLATES: tuple[str, ...] = (
    "Start with {anchor} at {anchor_coord} and propagate the coordinates along the chain.",
    "Begin from the known point {anchor} = {anchor_coord}.",
    "Use {anchor} at {anchor_coord} as the starting reference.",
    "The fixed starting coordinate is {anchor} at {anchor_coord}.",
    "We can solve the path step by step from {anchor}, whose coordinates are {anchor_coord}.",
    "Anchor the calculation at {anchor} = {anchor_coord}, then follow the relations in path order.",
    "Take {anchor} at {anchor_coord} as known and update the coordinate after each relation.",
    "The coordinate chain starts at {anchor}, located at {anchor_coord}.",
    "Use the known coordinate {anchor} = {anchor_coord} as the basis for the calculation.",
    "Take {anchor}'s coordinate, {anchor_coord}, as the fixed reference point.",
    "The calculation begins at the known location of {anchor}, namely {anchor_coord}.",
    "Set {anchor} at its given coordinate {anchor_coord} and work outward from there.",
    "Treat {anchor} = {anchor_coord} as the known point from which the coordinates will be propagated.",
    "The known position is {anchor} at {anchor_coord}; use it to start the coordinate updates.",
    "Begin the coordinate propagation from {anchor}, which is fixed at {anchor_coord}.",
    "Take the supplied position of {anchor}, {anchor_coord}, as the starting point.",
    "Fix {anchor} at {anchor_coord} before evaluating the spatial relations.",
    "Use {anchor}'s known location {anchor_coord} to initialize the coordinate calculation.",
    "Start the calculation from the given anchor point {anchor} at {anchor_coord}.",
    "The reference coordinate for the calculation is {anchor} = {anchor_coord}.",
    "With {anchor} fixed at {anchor_coord}, propagate positions through the relevant relations.",
    "Initialize the spatial calculation at {anchor} = {anchor_coord}.",
    "Begin with the provided coordinate for {anchor}, {anchor_coord}, and determine the remaining positions from it.",
    "Use the stated location of {anchor}, {anchor_coord}, as the fixed coordinate reference.",
    "We know {anchor} is at {anchor_coord}; start the coordinate reasoning from that point.",
    "Take {anchor}'s position to be {anchor_coord} and propagate the coordinates from there.",
    "The coordinate reasoning is anchored at {anchor}, whose known position is {anchor_coord}.",
    "Start from {anchor}'s given location, {anchor_coord}, and evaluate the connected relations from there.",
)

# The problem statements are deliberately shuffled.  A complete reference trace
# should therefore verbalize the topological reasoning that identifies the
# anchor-to-target chain before it starts doing coordinate arithmetic.  This
# line is semantic-world invariant: ordinary, fixed-R90, and rotating-frame
# references share exactly the same entity chain.
SOLUTION_PATH_TEMPLATES: tuple[str, ...] = (
    # Arrow-style formulations.
    "First, reconstruct the path from the known point to the target: {chain_arrow}.",
    "The entity chain to follow is {chain_arrow}.",
    "Before updating coordinates, trace the relations in this order: {chain_arrow}.",
    "Reading the relations in path order gives {chain_arrow}.",
    "The connected path from the known point to the requested entity is {chain_arrow}.",
    "The relevant relations connect the entities in the order {chain_arrow}.",
    "Putting the shuffled relations into traversal order yields {chain_arrow}.",
    "Reordering the relations from the anchor to the target gives {chain_arrow}.",
    "For coordinate propagation, the correct entity order is {chain_arrow}.",
    "The shuffled statements resolve into the path {chain_arrow}.",
    "Once the shuffled links are ordered, the traversal is {chain_arrow}.",
    "The route implied by the spatial relations is {chain_arrow}.",
    "The relation graph gives the ordered path {chain_arrow}.",
    "The coordinate updates should therefore proceed along {chain_arrow}.",
    "Resolve the shuffled statements into the traversal {chain_arrow} before updating coordinates.",
    # Prose-list formulations avoid making the arrow notation itself a fixed
    # answer-side signature.
    "The relevant sequence of entities is {chain_list}.",
    "The route can be read as the ordered entity sequence {chain_list}.",
    "The entities on the required route, in order, are {chain_list}.",
    "Following the connected relations takes us through {chain_list}.",
    "The anchor-to-target route visits {chain_list}, in that order.",
    "The relation chain consists of {chain_list} in traversal order.",
    "To propagate the coordinates correctly, use the ordered sequence {chain_list}.",
    "The path-relevant entities, ordered from the known point to the target, are {chain_list}.",
    "Tracing only the connected route gives the entity order {chain_list}.",
    "The spatial links establish this ordered route: {chain_list}.",
    # Sequential prose formulations make the planning step read more like
    # ordinary reasoning than a metadata declaration.
    "Starting from the known point, proceed to {chain_then}.",
    "Following the relations in sequence takes us from {anchor} to {chain_then}.",
    "To reach {target}, move through the linked entities as follows: {chain_then}.",
    "Tracing the route forward from {anchor}, we visit {chain_then}.",
    "The connected relations lead from the anchor to {chain_then}.",
    "Reading the shuffled statements as one continuous route gives: {chain_then}.",
    "The route can be followed naturally from {anchor}: {chain_then}.",
    "Before doing the coordinate arithmetic, follow the links from {anchor} to {chain_then}.",
    "The graph connectivity shows that the calculation should advance to {chain_then}.",
    "From the known entity, the path progresses to {chain_then}.",
    # Compact formulations.
    "The required traversal is {chain_arrow}.",
    "The anchor-to-target chain is {chain_arrow}.",
    "The path to evaluate is {chain_arrow}.",
    "The linked relations establish the route {chain_arrow}.",
    "The relevant path through the relation graph is {chain_arrow}.",
)

DIRECT_RELATION_TEMPLATES: tuple[str, ...] = (
    "Step {step}: The next relation already runs from {parent} to {child}.",
    "Step {step}: Follow the stated relation directly from {parent} to {child}.",
    "Step {step}: No inversion is needed for the link {parent} -> {child}.",
    "Step {step}: Propagate the position directly from {parent} to {child}.",
    "Step {step}: The given relation has the same orientation as the path from {parent} to {child}.",
    "Step {step}: The stated relation already points from {parent} toward {child}, so use it as written.",
    "Step {step}: This relation is aligned with our traversal from {parent} to {child}.",
    "Step {step}: The relation is already oriented correctly for moving from {parent} to {child}.",
    "Step {step}: Continue directly from {parent} to {child}; the stated relation needs no reversal.",
    "Step {step}: The statement gives the relation in the same direction we need, from {parent} to {child}.",
    "Step {step}: Use the relation as stated to advance from {parent} to {child}.",
    "Step {step}: The link is already written in traversal order, {parent} to {child}.",
    "Step {step}: From {parent}, the stated relation leads directly to {child}.",
    "Step {step}: The relation matches the direction of travel from {parent} to {child}.",
    "Step {step}: Keep the stated orientation and move from {parent} to {child}.",
    "Step {step}: The next link is already expressed in the forward direction, from {parent} to {child}.",
    "Step {step}: No relation inversion is required here; proceed from {parent} to {child}.",
    "Step {step}: The statement can be applied directly when going from {parent} to {child}.",
    "Step {step}: The stated orientation agrees with the required path from {parent} to {child}.",
    "Step {step}: Read this relation in its given direction, taking us from {parent} to {child}.",
    "Step {step}: This edge already follows the anchor-to-target traversal from {parent} to {child}.",
    "Step {step}: The relation needs no adjustment before propagating from {parent} to {child}.",
    "Step {step}: We can use this relation directly to carry the coordinates from {parent} to {child}.",
    "Step {step}: The relation is stated in the needed direction, with {parent} preceding {child} on the path.",
    "Step {step}: Proceed from {parent} to {child} using the relation exactly as given.",
    "Step {step}: The next connection has the correct orientation for moving from {parent} to {child}.",
    "Step {step}: The path direction and the stated relation coincide here, from {parent} to {child}.",
    "Step {step}: Apply the stated relation without inversion to move from {parent} to {child}.",
    "Step {step}: This link already faces the way we are traversing it, from {parent} to {child}.",
    "Step {step}: The relation is forward with respect to the required path, so continue from {parent} to {child}.",
)

INVERSE_RELATION_TEMPLATES: tuple[str, ...] = (
    "Step {step}: The relation is stated from {child} back to {parent}, so invert it to follow {parent} -> {child}.",
    "Step {step}: This statement is reversed relative to our path; read it in the opposite direction to go from {parent} to {child}.",
    "Step {step}: To continue from {parent} to {child}, reverse the direction of the stated relation.",
    "Step {step}: The problem describes {parent} relative to {child}; invert that relation before advancing from {parent} to {child}.",
    "Step {step}: The edge is written backward for our chain, so use its opposite direction from {parent} to {child}.",
    "Step {step}: The stated relation points from {child} toward {parent}; reverse it before moving from {parent} to {child}.",
    "Step {step}: This link is oriented opposite to our traversal, so invert it to proceed from {parent} to {child}.",
    "Step {step}: The relation is given in the reverse orientation; use its inverse for the move from {parent} to {child}.",
    "Step {step}: To traverse this edge from {parent} to {child}, take the opposite of the stated relation.",
    "Step {step}: The statement runs against the direction we need, so reverse the relation before going from {parent} to {child}.",
    "Step {step}: Here the relation is written from {child} back toward {parent}; invert it for the forward path.",
    "Step {step}: The next connection is backward relative to the route, so reverse its direction to move from {parent} to {child}.",
    "Step {step}: Read the stated relation in reverse so that the traversal goes from {parent} to {child}.",
    "Step {step}: The problem gives the opposite orientation for this link; invert it before propagating from {parent} to {child}.",
    "Step {step}: Because this relation is stated backward, use the opposite direction when moving from {parent} to {child}.",
    "Step {step}: The relation must be inverted here, since our path goes from {parent} to {child} rather than the other way around.",
    "Step {step}: This edge faces the reverse way along the chain; flip the relation to continue from {parent} to {child}.",
    "Step {step}: The stated direction corresponds to {child} -> {parent}, so use the inverse relation for {parent} -> {child}.",
    "Step {step}: The link is described in reverse path order; convert it to the opposite direction before advancing to {child}.",
    "Step {step}: Our traversal goes from {parent} to {child}, opposite to the statement's orientation, so invert the relation.",
    "Step {step}: Reverse this relation before using it, because the chain proceeds from {parent} to {child}.",
    "Step {step}: The statement gives {parent} relative to {child}; use the inverse relation to determine {child} from {parent}.",
    "Step {step}: This connection is expressed backward with respect to the route, so take its opposite before proceeding.",
    "Step {step}: To keep following the chain from {parent} to {child}, reinterpret the stated relation in the opposite direction.",
    "Step {step}: The relation's written orientation is reversed here; flip it before carrying the coordinates from {parent} to {child}.",
    "Step {step}: Since the statement runs from {child} toward {parent}, invert it to recover the move from {parent} to {child}.",
    "Step {step}: This relation points backward along the required path, so use its opposite orientation to reach {child} from {parent}.",
    "Step {step}: The edge is specified in reverse, so invert the relation before traversing it from {parent} to {child}.",
    "Step {step}: The stated link goes the other way around; reverse it before updating the position from {parent} to {child}.",
    "Step {step}: The path requires the inverse of the stated relation here, allowing us to move from {parent} to {child}.",
)

SOLUTION_STEP_FAMILIES: tuple[SolutionStepFamily, ...] = (
    SolutionStepFamily("move_then_disp", MOTION_ACTION, "narrative", "From {parent} at {parent_coord}, move {amount} {phrase}. This gives displacement {disp}, so {child} is at {child_coord}."),
    SolutionStepFamily("move_add", MOTION_ACTION, "coordinate_update", "Starting at {parent_coord}, a move of {amount} {phrase} adds {disp}. Therefore {child} is at {child_coord}."),
    SolutionStepFamily("leg_equation", MOTION_ACTION, "equation", "The leg from {parent} to {child} goes {amount} {phrase}, corresponding to {disp}. Numerically, {parent_coord} + {disp} = {child_coord}."),
    SolutionStepFamily("travel_place", MOTION_ACTION, "narrative", "{parent} is at {parent_coord}. Travel {amount} {phrase}; applying {disp} places {child} at {child_coord}."),
    SolutionStepFamily("go_add", MOTION_ACTION, "coordinate_update", "For this step, go {amount} {phrase}. Adding {disp} to {parent_coord} gives {child_coord} for {child}."),
    SolutionStepFamily("direction_update", MOTION_ACTION, "displacement_first", "A movement of {amount} {phrase} corresponds to {disp}. Applied to {parent}'s coordinate {parent_coord}, it puts {child} at {child_coord}."),
    SolutionStepFamily("path_update", MOTION_ACTION, "route_description", "From {parent}, follow the path {amount} {phrase}. The coordinate update is {parent_coord} + {disp} = {child_coord}, so this is {child}'s position."),
    SolutionStepFamily("compact_update", MOTION_ACTION, "compact", "Move {amount} {phrase} from {parent_coord}: add {disp} to obtain {child} at {child_coord}."),
    SolutionStepFamily("turn_then_disp", TURN_ACTION, "oriented_turn", "From {parent} at {parent_coord}, {phrase} for {amount}. The resulting displacement is {disp}, placing {child} at {child_coord}."),
    SolutionStepFamily("turn_equation", TURN_ACTION, "oriented_turn", "At {parent_coord}, {phrase} for {amount}. This is a displacement of {disp}, so {parent_coord} + {disp} = {child_coord} for {child}."),
    SolutionStepFamily("turn_route", TURN_ACTION, "oriented_turn", "To travel from {parent} to {child}, {phrase} for {amount}. Applying {disp} to {parent_coord} gives {child_coord}."),
    SolutionStepFamily("turn_compact", TURN_ACTION, "oriented_turn", "From {parent}'s position {parent_coord}, {phrase} for {amount}; the update {disp} puts {child} at {child_coord}."),
)

SOLUTION_FINAL_TEMPLATES: tuple[str, ...] = (
    "Therefore, the coordinates of {target} are \\boxed{{{final_coord}}}.",
    "Thus, {target} is located at \\boxed{{{final_coord}}}.",
    "Hence the coordinate pair for {target} is \\boxed{{{final_coord}}}.",
    "So the final coordinates of {target} are \\boxed{{{final_coord}}}.",
    "This places {target} at \\boxed{{{final_coord}}}.",
    "The chain therefore ends with {target} at \\boxed{{{final_coord}}}.",
    "Consequently, {target}'s coordinates are \\boxed{{{final_coord}}}.",
    "The requested coordinate pair is \\boxed{{{final_coord}}}.",
)


@dataclass(frozen=True)
class EdgeRecord:
    """One backbone edge from parent to child in query-path order."""

    parent: str
    child: str
    path_direction: str
    distance: int
    statement_reversed: bool
    statement_direction: str
    statement_subject: str
    statement_reference: str
    statement_text: str
    statement_family_id: int
    statement_family_key: str
    statement_syntax_family: str
    phrase_id: int
    phrase_kind: str
    phrase_semantic_family: str
    phrase_grammar_role: str
    amount_text: str


@dataclass(frozen=True)
class CoordinateVerificationResult:
    correct: bool
    extracted_prediction: str = ""
    parsed_prediction: tuple[int, int] | None = None
    parsed_gold: tuple[int, int] | None = None
    error: str = ""
    method: str = "strict_boxed_integer_coordinate"


def stable_hash(*parts: Any, length: int = 24) -> str:
    payload = "\n".join(
        json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        if not isinstance(part, str)
        else part
        for part in parts
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def stable_int(*parts: Any, bits: int = 63) -> int:
    return int(stable_hash(*parts, length=16), 16) & ((1 << bits) - 1)


def rotate_r90(vector: tuple[int, int]) -> tuple[int, int]:
    """Rotate a vector 90 degrees clockwise: (x, y) -> (y, -x)."""

    x, y = vector
    return y, -x


def rotate_quarter_turns(
    vector: tuple[int, int],
    turns_clockwise: int,
) -> tuple[int, int]:
    """Rotate ``vector`` by an integer number of clockwise quarter-turns.

    This helper is intentionally general and leaves the existing fixed-R90
    semantics unchanged.  It is used by the temporary rotating-frame ablation,
    where hop ``i`` applies ``i`` clockwise quarter-turns (modulo four).
    """

    turns = int(turns_clockwise) % 4
    result = vector
    for _ in range(turns):
        result = rotate_r90(result)
    return result


def vector_for_direction(direction: str, *, world: str) -> tuple[int, int]:
    if direction not in DIRECTION_SPECS:
        raise KeyError(f"Unknown direction: {direction!r}")
    ordinary = DIRECTION_SPECS[direction].ordinary_vector
    if world == "normal":
        return ordinary
    if world in {"r90", "alternative", "contradiction"}:
        return rotate_r90(ordinary)
    raise ValueError(f"Unknown world {world!r}; expected 'normal' or 'r90'.")


def scaled(vector: tuple[int, int], distance: int) -> tuple[int, int]:
    return vector[0] * int(distance), vector[1] * int(distance)


def add_coord(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def format_coord(coord: tuple[int, int]) -> str:
    return f"({coord[0]}, {coord[1]})"


def format_displacement(vector: tuple[int, int]) -> str:
    def f(value: int) -> str:
        return f"+{value}" if value > 0 else str(value)

    return f"({f(vector[0])}, {f(vector[1])})"


def render_amount(distance: int, rng: random.Random) -> str:
    """Render an exact grid distance with controlled lexical variation."""

    if distance not in NUMBER_WORDS:
        raise ValueError(f"Unsupported distance {distance}; expected 1..4")

    if distance == 1:
        number = rng.choices(("one", "1", "a single"), weights=(0.50, 0.30, 0.20), k=1)[0]
    else:
        number = NUMBER_WORDS[distance] if rng.random() < 0.58 else str(distance)

    unit_family = rng.choices(
        population=("cell", "grid cell", "step", "square", "grid square"),
        weights=(0.46, 0.24, 0.12, 0.10, 0.08),
        k=1,
    )[0]
    unit = unit_family if distance == 1 else unit_family + "s"
    return f"{number} {unit}"


def _phrase_candidates(
    spec: DirectionSpec,
    phrase_kind: str,
    phrase_role: str,
    allowed_semantic_families: frozenset[str] | None,
    semantic_family: str | None = None,
) -> tuple[PhraseSpec, ...]:
    """Return phrases licensed for a precise grammar role and family filter."""

    phrases = spec.relative_phrases if phrase_kind == "relative" else spec.motion_phrases
    candidates = tuple(
        phrase
        for phrase in phrases
        if phrase_role in phrase.grammar_roles
        and (allowed_semantic_families is None or phrase.semantic_family in allowed_semantic_families)
        and (semantic_family is None or phrase.semantic_family == semantic_family)
    )
    return candidates


def _available_statement_semantic_families(
    spec: DirectionSpec,
    allowed_semantic_families: frozenset[str] | None,
) -> tuple[str, ...]:
    families: set[str] = set()
    for statement_family in STATEMENT_FAMILIES:
        for phrase in _phrase_candidates(
            spec,
            statement_family.phrase_kind,
            statement_family.phrase_role,
            allowed_semantic_families,
        ):
            families.add(phrase.semantic_family)
    return tuple(sorted(families))


def _sample_phrase_for_role(
    spec: DirectionSpec,
    *,
    phrase_kind: str,
    phrase_role: str,
    semantic_family: str,
    rng: random.Random,
    allowed_semantic_families: frozenset[str] | None,
) -> tuple[PhraseSpec, int]:
    all_phrases = spec.relative_phrases if phrase_kind == "relative" else spec.motion_phrases
    candidates = _phrase_candidates(
        spec,
        phrase_kind,
        phrase_role,
        allowed_semantic_families,
        semantic_family,
    )
    if not candidates:
        raise ValueError(
            f"No {phrase_kind}/{phrase_role} phrases remain for direction {spec.key!r} "
            f"and semantic family {semantic_family!r}"
        )
    selected = rng.choice(candidates)
    return selected, all_phrases.index(selected)


def render_statement(
    *,
    subject: str,
    reference: str,
    direction: str,
    distance: int,
    rng: random.Random,
    allowed_semantic_families: frozenset[str] | None = None,
) -> RenderedStatement:
    """Render one exact, natural, lexically diverse relation statement.

    Sampling is semantic-family first, then grammar template, then phrase. This
    keeps families balanced while the explicit grammar-role compatibility layer
    prevents semantically valid phrases from entering unnatural syntactic slots.
    """

    spec = DIRECTION_SPECS[direction]
    semantic_families = _available_statement_semantic_families(spec, allowed_semantic_families)
    if not semantic_families:
        raise ValueError(
            f"No statement realization remains for direction {direction!r} under "
            f"semantic-family filter {sorted(allowed_semantic_families or ())}"
        )
    semantic_family = rng.choice(semantic_families)

    compatible: list[tuple[int, StatementFamily]] = []
    for family_id, family in enumerate(STATEMENT_FAMILIES):
        if _phrase_candidates(
            spec,
            family.phrase_kind,
            family.phrase_role,
            allowed_semantic_families,
            semantic_family,
        ):
            compatible.append((family_id, family))
    if not compatible:
        raise RuntimeError(f"Internal grammar coverage failure for {direction}/{semantic_family}")

    chosen_index = rng.choices(
        range(len(compatible)),
        weights=[family.weight for _, family in compatible],
        k=1,
    )[0]
    family_id, family = compatible[chosen_index]
    phrase, phrase_id = _sample_phrase_for_role(
        spec,
        phrase_kind=family.phrase_kind,
        phrase_role=family.phrase_role,
        semantic_family=semantic_family,
        rng=rng,
        allowed_semantic_families=allowed_semantic_families,
    )
    amount = render_amount(distance, rng)
    text = family.template.format(
        subject=subject,
        reference=reference,
        amount=amount,
        amount_cap=capitalize_sentence(amount),
        phrase=phrase.text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return RenderedStatement(
        text=text,
        family_id=family_id,
        family_key=family.key,
        syntax_family=family.syntax_family,
        phrase_id=phrase_id,
        phrase_kind=family.phrase_kind,
        phrase_semantic_family=phrase.semantic_family,
        phrase_grammar_role=family.phrase_role,
        amount_text=amount,
    )

def canonical_amount(distance: int) -> str:
    """Return a compact, natural amount phrase for reference reasoning."""

    if distance not in NUMBER_WORDS:
        raise ValueError(f"Unsupported distance {distance}; expected 1..4")
    number = NUMBER_WORDS[distance]
    unit = "cell" if distance == 1 else "cells"
    return f"{number} {unit}"


def canonical_relation_clause(subject: str, reference: str, direction: str, distance: int) -> str:
    amount = canonical_amount(distance)
    phrase = DIRECTION_SPECS[direction].canonical_phrase
    return f"{subject} is {amount} {phrase} {reference}"


def canonical_movement_clause(direction: str, distance: int) -> str:
    """Return e.g. ``one cell east`` or ``three cells northwest``."""

    if direction not in CANONICAL_MOTION_NAMES:
        raise KeyError(f"Unknown direction: {direction!r}")
    return f"{canonical_amount(distance)} {CANONICAL_MOTION_NAMES[direction]}"


def capitalize_sentence(text: str) -> str:
    """Uppercase only the first character without lowercasing proper names."""

    return text[:1].upper() + text[1:] if text else text


def build_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": OPENR1_SPATIAL_PROMPT_TEMPLATE.format(problem=problem.strip()),
        }
    ]


def render_problem(
    *,
    anchor: str,
    anchor_coord: tuple[int, int] = (0, 0),
    target: str,
    statements: Sequence[str],
    rng: random.Random,
) -> tuple[str, int, int]:
    """Render a problem without assuming the anchor is the global origin."""

    anchor_id = rng.randrange(len(ANCHOR_TEMPLATES))
    query_id = rng.randrange(len(QUERY_TEMPLATES))
    anchor_text = ANCHOR_TEMPLATES[anchor_id].format(
        anchor=anchor,
        anchor_coord=format_coord(anchor_coord),
    )
    lines = [anchor_text, "", *statements, "", QUERY_TEMPLATES[query_id].format(anchor=anchor, target=target)]
    return "\n".join(lines), anchor_id, query_id


def edge_coordinate_trace(
    anchor: str,
    edges: Sequence[EdgeRecord | Mapping[str, Any]],
    *,
    world: str,
    anchor_coord: tuple[int, int] = (0, 0),
) -> list[dict[str, Any]]:
    """Compute the exact coordinate trace along the backbone query path.

    ``world="rotating_r90"`` is an optional ablation mode.  It does not alter
    the canonical normal or fixed-R90 benchmark behavior.
    """

    current_entity = anchor
    current = tuple(anchor_coord)
    trace: list[dict[str, Any]] = []

    for step_id, edge_like in enumerate(edges, start=1):
        edge = edge_like if isinstance(edge_like, EdgeRecord) else EdgeRecord(**dict(edge_like))
        if edge.parent != current_entity:
            raise ValueError(
                f"Broken path at step {step_id}: expected parent {current_entity!r}, "
                f"found {edge.parent!r}"
            )
        if world in {"rotating_r90", "rotating", "stateful_r90"}:
            # Stateful ablation: the first traversed relation is interpreted in
            # an R90 frame, the second in R180, the third in R270, the fourth
            # in the ordinary frame, then the cycle repeats.  State follows
            # reasoning-path order, never shuffled statement order.
            ordinary = DIRECTION_SPECS[edge.path_direction].ordinary_vector
            quarter_turns = step_id % 4
            displacement = scaled(
                rotate_quarter_turns(ordinary, quarter_turns),
                edge.distance,
            )
        else:
            quarter_turns = None
            displacement = scaled(
                vector_for_direction(edge.path_direction, world=world),
                edge.distance,
            )
        new_coord = add_coord(current, displacement)
        trace.append(
            {
                "step": step_id,
                "parent": edge.parent,
                "child": edge.child,
                "direction": edge.path_direction,
                "distance": edge.distance,
                "parent_coord": list(current),
                "displacement": list(displacement),
                "child_coord": list(new_coord),
                "statement_reversed": edge.statement_reversed,
                "statement_text": edge.statement_text,
                "frame_quarter_turns": quarter_turns,
            }
        )
        current_entity = edge.child
        current = new_coord

    return trace


def _available_solution_semantic_families(
    spec: DirectionSpec,
    allowed_semantic_families: frozenset[str] | None,
) -> tuple[str, ...]:
    families: set[str] = set()
    for step_family in SOLUTION_STEP_FAMILIES:
        for phrase in _phrase_candidates(
            spec,
            "motion",
            step_family.phrase_role,
            allowed_semantic_families,
        ):
            families.add(phrase.semantic_family)
    return tuple(sorted(families))


def render_solution(
    *,
    anchor: str,
    anchor_coord: tuple[int, int] = (0, 0),
    target: str,
    edges: Sequence[EdgeRecord | Mapping[str, Any]],
    world: str,
    solution_seed: int,
    allowed_semantic_families: frozenset[str] | None = None,
) -> tuple[str, tuple[int, int], list[dict[str, Any]]]:
    """Render a deterministic, exact, richly varied SFT reference solution.

    The answer-side direction language uses the same semantic-family inventory
    as the questions. Family-first sampling prevents the target responses from
    collapsing back to a small compass-only template set. The same seed and
    family filter are used in both worlds, so paired ordinary/R90 responses have
    identical wording choices and differ only in the coordinate semantics.
    """

    rng = random.Random(solution_seed)
    edge_objs = [edge if isinstance(edge, EdgeRecord) else EdgeRecord(**dict(edge)) for edge in edges]
    trace = edge_coordinate_trace(anchor, edge_objs, world=world, anchor_coord=anchor_coord)
    lines = [
        rng.choice(SOLUTION_START_TEMPLATES).format(
            anchor=anchor,
            anchor_coord=format_coord(anchor_coord),
        )
    ]

    # Path reconstruction is a genuine reasoning step only when multiple
    # shuffled relations must be ordered.  For a one-hop problem there is no
    # path ambiguity to resolve, so omit the planning line to avoid repetitive
    # boilerplate such as "The entity chain to follow is Ari -> Bela."
    # Use an independent deterministic RNG so this explanatory line does not
    # perturb the existing answer-side phrase/template choices.
    if len(edge_objs) > 1:
        path_entities = [anchor, *(edge.child for edge in edge_objs)]
        chain_arrow = " -> ".join(path_entities)
        if len(path_entities) == 2:
            chain_list = " and ".join(path_entities)
            chain_then = path_entities[1]
        else:
            chain_list = ", ".join(path_entities[:-1]) + f", and {path_entities[-1]}"
            intermediate = path_entities[1:-1]
            if len(intermediate) == 1:
                chain_then = f"{intermediate[0]}, then {path_entities[-1]}"
            else:
                chain_then = ", then ".join(intermediate) + f", and finally {path_entities[-1]}"

        planning_rng = random.Random(stable_int(solution_seed, "path_reconstruction"))
        lines.append(
            planning_rng.choice(SOLUTION_PATH_TEMPLATES).format(
                chain_arrow=chain_arrow,
                chain_list=chain_list,
                chain_then=chain_then,
                anchor=anchor,
                target=target,
                anchor_coord=format_coord(anchor_coord),
            )
        )

    for item, edge in zip(trace, edge_objs):
        parent_coord = tuple(item["parent_coord"])
        child_coord = tuple(item["child_coord"])
        disp = tuple(item["displacement"])

        if edge.statement_reversed:
            transition = rng.choice(INVERSE_RELATION_TEMPLATES).format(
                step=item["step"],
                parent=edge.parent,
                child=edge.child,
            )
        else:
            transition = rng.choice(DIRECT_RELATION_TEMPLATES).format(
                step=item["step"],
                parent=edge.parent,
                child=edge.child,
            )
        lines.append(transition)

        spec = DIRECTION_SPECS[edge.path_direction]
        semantic_families = _available_solution_semantic_families(spec, allowed_semantic_families)
        if not semantic_families:
            raise ValueError(
                f"No solution realization remains for direction {edge.path_direction!r} under "
                f"semantic-family filter {sorted(allowed_semantic_families or ())}"
            )
        semantic_family = rng.choice(semantic_families)
        compatible: list[tuple[int, SolutionStepFamily]] = []
        for family_id, family in enumerate(SOLUTION_STEP_FAMILIES):
            if _phrase_candidates(
                spec,
                "motion",
                family.phrase_role,
                allowed_semantic_families,
                semantic_family,
            ):
                compatible.append((family_id, family))
        chosen_index = rng.choices(
            range(len(compatible)),
            weights=[family.weight for _, family in compatible],
            k=1,
        )[0]
        family_id, family = compatible[chosen_index]
        phrase, phrase_id = _sample_phrase_for_role(
            spec,
            phrase_kind="motion",
            phrase_role=family.phrase_role,
            semantic_family=semantic_family,
            rng=rng,
            allowed_semantic_families=allowed_semantic_families,
        )
        amount = render_amount(edge.distance, rng)
        step_text = family.template.format(
            parent=edge.parent,
            child=edge.child,
            parent_coord=format_coord(parent_coord),
            child_coord=format_coord(child_coord),
            disp=format_displacement(disp),
            amount=amount,
            amount_cap=capitalize_sentence(amount),
            phrase=phrase.text,
            phrase_cap=capitalize_sentence(phrase.text),
        )
        lines.append(re.sub(r"\s+", " ", step_text).strip())

        # Retain answer-side surface provenance in the structured trace. This
        # enables explicit SFT-target diversity analysis without bloating the
        # clean model-facing text.
        item["solution_phrase_text"] = phrase.text
        item["solution_phrase_semantic_family"] = phrase.semantic_family
        item["solution_phrase_grammar_role"] = family.phrase_role
        item["solution_phrase_id"] = phrase_id
        item["solution_step_family_id"] = family_id
        item["solution_step_family_key"] = family.key
        item["solution_step_syntax_family"] = family.syntax_family
        item["solution_amount_text"] = amount

    final_coord = tuple(trace[-1]["child_coord"]) if trace else tuple(anchor_coord)
    lines.append(
        rng.choice(SOLUTION_FINAL_TEMPLATES).format(
            target=target,
            final_coord=format_coord(final_coord),
        )
    )
    return "\n".join(lines), final_coord, trace

def _extract_last_boxed(text: Any) -> str | None:
    raw = str(text or "")
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", raw)]
    last: str | None = None
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
                    last = raw[brace_start + 1 : pos].strip()
                    break
    return last


def parse_coordinate(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace("−", "-").strip().strip("$")
    match = re.fullmatch(
        r"\(?\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)?",
        text,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


class SpatialCoordinateVerifier:
    """Exact evaluator for boxed integer coordinate pairs."""

    def verify_with_details(self, gold: Any, prediction: Any) -> CoordinateVerificationResult:
        gold_coord = parse_coordinate(gold)
        if gold_coord is None:
            return CoordinateVerificationResult(
                correct=False,
                parsed_gold=None,
                error=f"invalid gold coordinate: {gold!r}",
            )

        boxed = _extract_last_boxed(prediction)
        if boxed is None:
            return CoordinateVerificationResult(
                correct=False,
                parsed_gold=gold_coord,
                error="no boxed coordinate found",
            )

        predicted = parse_coordinate(boxed)
        if predicted is None:
            return CoordinateVerificationResult(
                correct=False,
                extracted_prediction=boxed,
                parsed_gold=gold_coord,
                error="boxed content is not an integer coordinate pair",
            )

        return CoordinateVerificationResult(
            correct=predicted == gold_coord,
            extracted_prediction=boxed,
            parsed_prediction=predicted,
            parsed_gold=gold_coord,
        )

    def score_many(self, responses: Sequence[str], references: Sequence[Any]) -> list[int]:
        if len(responses) != len(references):
            raise ValueError(
                f"responses/references length mismatch: {len(responses)} != {len(references)}"
            )
        return [
            int(self.verify_with_details(reference, response).correct)
            for response, reference in zip(responses, references)
        ]


def direction_manifest() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in DIRECTION_SPECS.items():
        out[key] = {
            "ordinary_vector": list(spec.ordinary_vector),
            "r90_vector": list(rotate_r90(spec.ordinary_vector)),
            "opposite": spec.opposite,
            "relative_phrases": [p.text for p in spec.relative_phrases],
            "motion_phrases": [p.text for p in spec.motion_phrases],
            "relative_phrase_specs": [asdict(p) for p in spec.relative_phrases],
            "motion_phrase_specs": [asdict(p) for p in spec.motion_phrases],
            "canonical_phrase": spec.canonical_phrase,
        }
    return out


def vocabulary_manifest() -> dict[str, Any]:
    return {
        "version": VERSION,
        "transformation": TRANSFORMATION_NAME,
        "transformation_version": TRANSFORMATION_VERSION,
        "phrase_semantic_families": list(PHRASE_SEMANTIC_FAMILIES),
        "directions": direction_manifest(),
        "phrase_grammar_roles": list(PHRASE_GRAMMAR_ROLES),
        "statement_families": [asdict(family) for family in STATEMENT_FAMILIES],
        "solution_step_families": [asdict(family) for family in SOLUTION_STEP_FAMILIES],
        "anchor_templates": list(ANCHOR_TEMPLATES),
        "query_templates": list(QUERY_TEMPLATES),
        "solution_start_templates": list(SOLUTION_START_TEMPLATES),
        "solution_path_templates": list(SOLUTION_PATH_TEMPLATES),
        "direct_relation_templates": list(DIRECT_RELATION_TEMPLATES),
        "inverse_relation_templates": list(INVERSE_RELATION_TEMPLATES),
        "solution_final_templates": list(SOLUTION_FINAL_TEMPLATES),
        "entity_names": list(ENTITY_NAMES),
        "distance_policy": {
            "supported": [1, 2, 3, 4],
            "default_weights": {"1": 0.48, "2": 0.30, "3": 0.16, "4": 0.06},
            "surface_units": ["cell", "grid cell", "step", "square", "grid square"],
            "diagonal_semantics": "distance d means d diagonal grid steps, i.e. magnitude d on both active axes",
        },
        "prompt_template": OPENR1_SPATIAL_PROMPT_TEMPLATE,
    }


def edge_to_dict(edge: EdgeRecord) -> dict[str, Any]:
    return asdict(edge)


def validate_phrase_family_coverage(
    allowed_semantic_families: frozenset[str] | None,
) -> list[str]:
    """Validate an optional phrase-family filter for questions and SFT targets."""

    if allowed_semantic_families is None:
        return []
    unknown = sorted(set(allowed_semantic_families) - set(PHRASE_SEMANTIC_FAMILIES))
    issues = [f"unknown phrase semantic families: {unknown}"] if unknown else []
    for key, spec in DIRECTION_SPECS.items():
        statement_available = _available_statement_semantic_families(spec, allowed_semantic_families)
        solution_available = _available_solution_semantic_families(spec, allowed_semantic_families)
        if not statement_available:
            issues.append(f"{key}: no question phrases remain under semantic-family filter")
        if not solution_available:
            issues.append(f"{key}: no solution phrases remain under semantic-family filter")
    return issues


def validate_vocabulary() -> list[str]:
    """Exhaustively validate vocabulary semantics and grammar compatibility."""

    issues: list[str] = []
    if set(DIRECTION_SPECS) != set(DIRECTION_KEYS):
        issues.append("direction key mismatch")

    statement_keys = [family.key for family in STATEMENT_FAMILIES]
    if len(statement_keys) != len(set(statement_keys)):
        issues.append("statement family keys must be unique")
    solution_keys = [family.key for family in SOLUTION_STEP_FAMILIES]
    if len(solution_keys) != len(set(solution_keys)):
        issues.append("solution step family keys must be unique")

    known_roles = set(PHRASE_GRAMMAR_ROLES)
    for key, spec in DIRECTION_SPECS.items():
        if spec.opposite not in DIRECTION_SPECS:
            issues.append(f"{key}: missing opposite {spec.opposite}")
        elif DIRECTION_SPECS[spec.opposite].opposite != key:
            issues.append(f"{key}: opposite relation is not symmetric")

        if not spec.relative_phrases or not spec.motion_phrases:
            issues.append(f"{key}: empty phrase inventory")

        seen_by_kind: dict[str, set[str]] = {"relative": set(), "motion": set()}
        for phrase_kind, phrases in (("relative", spec.relative_phrases), ("motion", spec.motion_phrases)):
            for phrase in phrases:
                normalized = phrase.text.strip().lower()
                if normalized in seen_by_kind[phrase_kind]:
                    issues.append(f"{key}: duplicate {phrase_kind} phrase {phrase.text!r}")
                seen_by_kind[phrase_kind].add(normalized)
                if phrase.semantic_family not in PHRASE_SEMANTIC_FAMILIES:
                    issues.append(f"{key}: unknown semantic family {phrase.semantic_family!r}")
                if not phrase.grammar_roles:
                    issues.append(f"{key}: phrase has no grammar role: {phrase.text!r}")
                unknown_roles = sorted(set(phrase.grammar_roles) - known_roles)
                if unknown_roles:
                    issues.append(f"{key}: phrase {phrase.text!r} has unknown roles {unknown_roles}")

        # Every phrase must be usable by at least one appropriate question
        # template; every motion phrase must also be usable in an SFT solution.
        for phrase_kind, phrases in (("relative", spec.relative_phrases), ("motion", spec.motion_phrases)):
            for phrase in phrases:
                statement_usable = any(
                    family.phrase_kind == phrase_kind and family.phrase_role in phrase.grammar_roles
                    for family in STATEMENT_FAMILIES
                )
                if not statement_usable:
                    issues.append(f"{key}: unused question phrase {phrase.text!r}")
                if phrase_kind == "motion":
                    solution_usable = any(
                        family.phrase_role in phrase.grammar_roles for family in SOLUTION_STEP_FAMILIES
                    )
                    if not solution_usable:
                        issues.append(f"{key}: motion phrase unavailable to solution renderer {phrase.text!r}")

        # Exhaustively render all licensed phrase/template combinations at both
        # singular and plural distances. This catches compatibility regressions.
        for distance, amount in ((1, "one cell"), (2, "two cells")):
            for family in STATEMENT_FAMILIES:
                phrases = _phrase_candidates(spec, family.phrase_kind, family.phrase_role, None)
                for phrase in phrases:
                    try:
                        rendered = family.template.format(
                            subject="Bela",
                            reference="Ari",
                            amount=amount,
                            amount_cap=capitalize_sentence(amount),
                            phrase=phrase.text,
                        )
                    except (KeyError, IndexError, ValueError) as exc:
                        issues.append(f"{key}/{family.key}: template formatting failed: {exc}")
                        continue
                    if "  " in rendered or not rendered.endswith("."):
                        issues.append(f"{key}/{family.key}: malformed rendering {rendered!r}")

            for family in SOLUTION_STEP_FAMILIES:
                phrases = _phrase_candidates(spec, "motion", family.phrase_role, None)
                for phrase in phrases:
                    try:
                        rendered = family.template.format(
                            parent="Ari",
                            child="Bela",
                            parent_coord="(0, 0)",
                            child_coord="(1, 0)",
                            disp="(+1, 0)",
                            amount=amount,
                            amount_cap=capitalize_sentence(amount),
                            phrase=phrase.text,
                            phrase_cap=capitalize_sentence(phrase.text),
                        )
                    except (KeyError, IndexError, ValueError) as exc:
                        issues.append(f"{key}/solution/{family.key}: formatting failed: {exc}")
                        continue
                    if "  " in rendered or not rendered.endswith("."):
                        issues.append(f"{key}/solution/{family.key}: malformed rendering {rendered!r}")

    # Vector and opposite invariants.
    for key, spec in DIRECTION_SPECS.items():
        ordinary_opp = DIRECTION_SPECS[spec.opposite].ordinary_vector
        if add_coord(spec.ordinary_vector, ordinary_opp) != (0, 0):
            issues.append(f"{key}: ordinary opposite vectors do not cancel")
        r90 = vector_for_direction(key, world="r90")
        r90_opp = vector_for_direction(spec.opposite, world="r90")
        if add_coord(r90, r90_opp) != (0, 0):
            issues.append(f"{key}: R90 opposite vectors do not cancel")

    # Validate the generated turn-composition semantics independently of text.
    order = CLOCKWISE_DIRECTION_ORDER
    for direction in DIRECTION_KEYS:
        index = order.index(direction)
        right_start = order[(index - 2) % len(order)]
        left_start = order[(index + 2) % len(order)]
        opposite_start = order[(index + 4) % len(order)]
        if order[(order.index(right_start) + 2) % len(order)] != direction:
            issues.append(f"{direction}: clockwise quarter-turn composition is incorrect")
        if order[(order.index(left_start) - 2) % len(order)] != direction:
            issues.append(f"{direction}: counterclockwise quarter-turn composition is incorrect")
        if order[(order.index(opposite_start) + 4) % len(order)] != direction:
            issues.append(f"{direction}: half-turn composition is incorrect")

    if set(CLOCK_LABELS) != set(DIRECTION_KEYS):
        issues.append("clock-face direction coverage mismatch")
    if set(COMPASS_BEARINGS) != set(DIRECTION_KEYS):
        issues.append("compass-bearing direction coverage mismatch")

    return issues