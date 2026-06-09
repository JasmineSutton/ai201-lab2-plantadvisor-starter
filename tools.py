import json
import os
from datetime import datetime
from config import DATA_PATH

# Plant database and seasonal data are loaded once at module load.
# This mirrors how a real service would cache its data source in memory.
with open(os.path.join(DATA_PATH, "plants.json"), encoding="utf-8") as f:
    _plant_db = json.load(f)

with open(os.path.join(DATA_PATH, "seasons.json"), encoding="utf-8") as f:
    _season_data = json.load(f)

# Maps calendar months to seasons for auto-detection.
_MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall",  10: "fall",  11: "fall",
}


def lookup_plant(plant_name: str) -> dict:
    """
    Search the plant database for a plant by name and return its care information.
    Handles direct key match, display name match, and alias match — all case-insensitive.
    """
    # Step 1: Normalize input — strip whitespace and lowercase everything
    normalized = plant_name.strip().lower()

    # Step 2: Direct key match — fastest lookup, O(1) dict access
    # Keys in plants.json are slugs like "pothos", "snake_plant"
    if normalized in _plant_db:
        return {"found": True, "plant": _plant_db[normalized]}

    # Step 3: Display name match — e.g. user typed "Pothos" or "Snake Plant"
    for key, plant in _plant_db.items():
        if plant["display_name"].lower() == normalized:
            return {"found": True, "plant": plant}

    # Step 4: Alias match — e.g. user typed "devil's ivy" or "mother-in-law's tongue"
    for key, plant in _plant_db.items():
        if normalized in [alias.lower() for alias in plant["aliases"]]:
            return {"found": True, "plant": plant}

    # Step 5: Nothing matched — return a helpful not-found message the LLM can use
    return {
        "found": False,
        "name": normalized,
        "message": (
            f"No plant matching '{plant_name}' was found in the database. "
            "The database contains common houseplants like pothos, monstera, snake plant, "
            "fiddle leaf fig, calathea, and others. "
            "Do not invent specific care instructions. "
            "Acknowledge that this plant is not in the database, and offer general guidance "
            "based on what the user describes about the plant (e.g., succulent, tropical, fern)."
        ),
    }


def get_seasonal_conditions(season: str | None = None) -> dict:
    """
    Return current seasonal care context for houseplants.

    If season is provided and valid, returns that season's data.
    If season is None (or invalid), auto-detects from the current calendar month.

    Pre-implemented — read through this and the spec before working on lookup_plant().
    """
    VALID_SEASONS = {"spring", "summer", "fall", "winter"}

    if season and season.lower() in VALID_SEASONS:
        # Caller specified a valid season — use it directly
        season_key = season.lower()
        detected = False
    else:
        # Auto-detect from the current month using the _MONTH_TO_SEASON mapping
        current_month = datetime.now().month
        season_key = _MONTH_TO_SEASON[current_month]
        detected = True

    # Copy the season dict so we don't mutate the cached data
    result = dict(_season_data[season_key])
    result["detected_season"] = detected
    return result
