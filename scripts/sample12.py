"""Shared source contract and measurement windows for the current sample 12."""

import re


FACTORS = {
    "temperature": ("    colony.temperature(0.85);\n", "    colony.temperature(0.0);\n"),
    "pitch_shift": ("    root.freq(root_hz * 1.5);\n", "    root.freq(root_hz);\n"),
    "flow": ("    let flow = place(flow_particles, consonance(200.0, 1500.0).count(9).spacing(0.66));\n",
             "    flow.amp(0.014);\n", "    release(flow);\n"),
}
COMMON_PLACEMENTS = (
    "    let root = place(field_anchor, at(root_hz).count(1));\n",
    "    let colony = place(consonance_colony, consonance(80.0, 900.0).count(8).spacing(0.84));\n",
)
RESERVE_RUNTIME_IDS_THROUGH = 18
COLONY_POPULATION_ID = 2
FLOW_START_SEC = 15.0
WAIT_SEQUENCE = ["2.3", "9.4", "3.3", "5.3", "3.3", "1.3", "5.3", "2.0", "4.0"]
OPERATION_WAIT_COUNTS = [
    *zip(COMMON_PLACEMENTS, (0, 1)),
    *zip(FACTORS["temperature"], (2, 4)),
    *zip(FACTORS["pitch_shift"], (2, 4)),
    *zip(FACTORS["flow"], (3, 4, 5)),
    ("    colony.amp(0.034);\n", 2),
    ('    colony.pitch_apply_mode("glide");\n', 4),
    ("    colony.glide(0.22);\n", 4),
    ("    colony.amp(0.028);\n", 6),
    ("    release(colony);\n", 7),
    ("    release(root);\n", 8),
]
WINDOWS = {"baseline": (2.3, 11.7), "tension": (15.0, 20.3),
           "early_resolution": (20.3, 23.6), "late_resolution": (24.9, 30.2)}
STANDARD_WINDOWS = {
    "baseline_colony": WINDOWS["baseline"],
    "tension_before_flow": (WINDOWS["baseline"][1], FLOW_START_SEC),
    "tension_with_flow": WINDOWS["tension"],
    "resolution_with_flow": WINDOWS["early_resolution"],
    "resolution_after_flow": WINDOWS["late_resolution"],
}


def validate_source(source):
    # Validate every intervention even when enabled, so the control cannot hide drift.
    for fragments in FACTORS.values():
        for fragment in fragments:
            if source.count(fragment) != 1:
                raise ValueError(f"sample 12 drift: expected exactly one {fragment.strip()!r}")
    # The ID reservation is tied to these three placements: 1 + 8 + 9.
    if (any(source.count(fragment) != 1 for fragment in COMMON_PLACEMENTS)
            or len(re.findall(r"\bplace\s*\(", source)) != 3):
        raise ValueError("sample 12 placements changed; review the runtime ID reservation")
    waits = re.findall(r"(?m)^\s*wait\(([^)]+)\);$", source)
    if waits != WAIT_SEQUENCE:
        raise ValueError("sample 12 wait sequence changed; review the measurement windows")
    for fragment, count in OPERATION_WAIT_COUNTS:
        if (source.count(fragment) != 1 or re.findall(r"(?m)^\s*wait\(([^)]+)\);$",
                                                     source[:source.index(fragment)]) != waits[:count]):
            raise ValueError(f"sample 12 operation timing changed: {fragment.strip()}")
