from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_scenario(path: Path) -> Dict[str, Any]:
    """
    Load a scenario.json file and return it as a dict.

    Expected structure (as in scenario.json):
      - horizon_days
      - policies
      - failure
      - repair
      - pm
      - costs
      - monte_carlo
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        scenario = json.load(f)

    return scenario
