"""Grid constants shared by every script in figures/.
"""

# task categories of the paper (Sec. 4)
CATEGORIES = {
    "non_contradictory": ["tooluse", "science", "spatial_standard2"],
    "contradictory": ["spatial_contradiction2", "math_contradiction"],  # spatial_contradiction (v1) is obsolete
    "novel_facts": ["fictionalqa"],
}
# paper display names (final layer only: data, filenames and columns keep the internal ids)
DISPLAY = {"tooluse": "tool-alpaca", "science": "chemistry", "spatial_standard2": "spatial",
           "spatial_contradiction2": "spatial-contradiction",
           "math_contradiction": "math-contradiction", "fictionalqa": "fictionalqa"}
# canonical training horizons per task (default epochs)
CANON_EP = {"tooluse": 2, "science": 2, "spatial_standard2": 4,
            "spatial_contradiction2": 4, "math_contradiction": 4}
