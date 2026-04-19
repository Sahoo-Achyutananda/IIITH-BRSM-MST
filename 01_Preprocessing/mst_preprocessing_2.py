"""Preprocessing pipeline for modified Mnemonic Similarity Task (MST).

This script pairs encoding ("_task_") and test ("_test_") files by participant ID,
extracts trial-level fields, and returns a combined dataframe across conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_rt(row: pd.Series) -> float:
    """Compute encoding RT for one row.

    Rule:
    - If `key_resp_9.rt` exists and is non-missing, use it.
    - Otherwise, RT = 3 + `key_resp_8.rt`.

    Missing values are returned as `np.nan`.
    """
    rt9 = pd.to_numeric(row.get("key_resp_9.rt"), errors="coerce")
    if pd.notna(rt9):
        return float(rt9)

    rt8 = pd.to_numeric(row.get("key_resp_8.rt"), errors="coerce")
    if pd.notna(rt8):
        return float(3 + rt8)

    return np.nan


def label_boundary_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add event and boundary labels based on trial index in 7-item events.

    Adds columns:
    - `event_number` = trial_index // 7
    - `position_in_event` = trial_index % 7
    - `boundary_position`:
        * 0 -> Post
        * 6 -> Pre
        * otherwise -> Mid
    """
    out = df.copy()
    out["event_number"] = out["trial_index"] // 7
    out["position_in_event"] = out["trial_index"] % 7

    out["boundary_position"] = np.where(
        out["position_in_event"] == 0,
        "Post",
        np.where(out["position_in_event"] == 6, "Pre", "Mid"),
    )
    return out


def classify_trial(path: object) -> Optional[str]:
    """Classify test trial type from an image path.

    Rules:
    - If path contains `foil/` (or `foil\\`) -> Foil
    - If path ends with `a.jpg` -> Target
    - If path ends with `b.jpg` -> Lure
    - Otherwise -> None
    """
    if pd.isna(path):
        return None

    normalized = str(path).strip().lower().replace("\\", "/")
    if "foil/" in normalized or "foils/" in normalized:
        return "Foil"
    if normalized.endswith("a.jpg"):
        return "Target"
    if normalized.endswith("b.jpg"):
        return "Lure"
    return None


def canonical_stimulus_key(path: object) -> Optional[str]:
    """Return a canonical key for joining encoding/test rows by stimulus identity.

    For MST object/scene items, lure files end in `b.jpg` while encoding targets
    end in `a.jpg`. We canonicalize both variants to the `a.jpg` form so lure
    test trials inherit boundary labels from their encoded target counterpart.

    Foils and non-standard paths are returned as normalized paths.
    """
    if pd.isna(path):
        return None

    normalized = str(path).strip().replace("\\", "/")
    lower = normalized.lower()

    if lower.endswith("b.jpg"):
        return f"{normalized[:-5]}a.jpg"
    return normalized


def _extract_response_key(df: pd.DataFrame) -> pd.Series:
    """Extract participant response key from the available test response columns."""
    candidate_columns = ["trials.key_resp_3.keys", "key_resp_3.keys", "key_resp.keys"]
    for column in candidate_columns:
        if column in df.columns:
            return df[column]
    return pd.Series([np.nan] * len(df), index=df.index)


def _extract_response_rt(df: pd.DataFrame) -> pd.Series:
    """Extract test-phase response time from available retrieval RT columns."""
    candidate_columns = ["trials.key_resp_3.rt", "key_resp_3.rt", "key_resp.rt"]
    for column in candidate_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)


def _map_response_to_label(response_key: object) -> Optional[str]:
    """Map raw response key to response category.

    Convention used in this MST version:
    - o -> Target ("old")
    - s -> Lure ("similar")
    - n -> Foil ("new")
    """
    if pd.isna(response_key):
        return None

    key = str(response_key).strip().lower()
    mapping = {
        "o": "Target",
        "s": "Lure",
        "n": "Foil",
    }
    return mapping.get(key)


def _participant_id_from_name(file_path: Path) -> Optional[str]:
    """Extract 5-digit participant ID from filename prefix."""
    prefix = file_path.name[:5]
    if prefix.isdigit() and len(prefix) == 5:
        return prefix
    return None


def process_participant(task_file: Path, test_file: Path, condition: str) -> pd.DataFrame:
    """Process one participant's task/test file pair into analysis-ready rows."""
    task_df = pd.read_csv(task_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)

    if "image_path" not in task_df.columns:
        raise ValueError(f"Missing 'image_path' in encoding file: {task_file}")
    if "image_path" not in test_df.columns:
        raise ValueError(f"Missing 'image_path' in test file: {test_file}")

    if "trials.thisN" in task_df.columns:
        encoding_trials = task_df[task_df["trials.thisN"].notna()].copy()
    else:
        encoding_trials = task_df.copy()

    encoding_trials = encoding_trials[encoding_trials["image_path"].notna()].copy()
    encoding_trials = encoding_trials[encoding_trials["image_path"].astype(str).str.strip() != ""]

    encoding_trials = encoding_trials.reset_index(drop=True)
    encoding_trials["trial_index"] = np.arange(len(encoding_trials), dtype=int)
    encoding_trials = label_boundary_position(encoding_trials)
    encoding_trials["RT"] = encoding_trials.apply(compute_rt, axis=1)

    boundary_lookup = encoding_trials[["image_path", "boundary_position", "RT"]].copy()
    boundary_lookup = boundary_lookup[boundary_lookup["image_path"].notna()]
    boundary_lookup["stimulus_key"] = boundary_lookup["image_path"].apply(canonical_stimulus_key)
    boundary_lookup = boundary_lookup[boundary_lookup["stimulus_key"].notna()]
    boundary_lookup = boundary_lookup.drop_duplicates(subset=["image_path"], keep="first")
    boundary_lookup = boundary_lookup.drop_duplicates(subset=["stimulus_key"], keep="first")

    test_trials = test_df[test_df["image_path"].notna()].copy()
    test_trials = test_trials[test_trials["image_path"].astype(str).str.strip() != ""]
    test_trials["trial_type"] = test_trials["image_path"].apply(classify_trial)
    test_trials["response_key"] = _extract_response_key(test_trials)
    test_trials["user_response"] = test_trials["response_key"].apply(_map_response_to_label)
    test_trials["stimulus_key"] = test_trials["image_path"].apply(canonical_stimulus_key)

    merged = test_trials.merge(
        boundary_lookup[["stimulus_key", "boundary_position", "RT"]],
        on="stimulus_key",
        how="left",
    )
    merged = merged.rename(columns={"RT": "encoding_RT"})
    merged["RT"] = _extract_response_rt(merged)

    participant_id = _participant_id_from_name(task_file)
    if participant_id is None:
        participant_id = "unknown"

    output = merged[
        ["image_path", "trial_type", "user_response", "boundary_position", "RT", "encoding_RT"]
    ].copy()
    output.insert(0, "participant_uid", f"{condition}_{participant_id}")
    output.insert(0, "condition", condition)
    output.insert(0, "participant_id", participant_id)
    return output


def process_condition(condition_folder_path: Path) -> pd.DataFrame:
    """Process all paired participants under one condition folder.

    The function searches recursively for CSV files, identifies encoding/test files,
    pairs by 5-digit participant ID, warns on missing pairs, and concatenates outputs.
    """
    condition_name = condition_folder_path.name
    csv_files = sorted(condition_folder_path.rglob("*.csv"))

    task_files: Dict[str, Path] = {}
    test_files: Dict[str, Path] = {}

    for csv_file in csv_files:
        participant_id = _participant_id_from_name(csv_file)
        if participant_id is None:
            continue

        lower_name = csv_file.name.lower()
        if "_task_" in lower_name:
            task_files[participant_id] = csv_file
        elif "_test_" in lower_name:
            test_files[participant_id] = csv_file

    all_ids = sorted(set(task_files).union(test_files))
    participant_frames: List[pd.DataFrame] = []

    for participant_id in all_ids:
        task_file = task_files.get(participant_id)
        test_file = test_files.get(participant_id)

        if task_file is None or test_file is None:
            print(
                f"Warning: pairing failed for participant {participant_id} in "
                f"{condition_name} (task: {task_file is not None}, test: {test_file is not None})."
            )
            continue

        try:
            participant_frames.append(
                process_participant(task_file=task_file, test_file=test_file, condition=condition_name)
            )
        except Exception as exc:
            print(
                f"Warning: failed processing participant {participant_id} in "
                f"{condition_name}: {exc}"
            )

    if not participant_frames:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "participant_uid",
                "condition",
                "image_path",
                "trial_type",
                "user_response",
                "boundary_position",
                "RT",
                "encoding_RT",
            ]
        )

    return pd.concat(participant_frames, ignore_index=True)


def main(base_data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run preprocessing across all conditions and return the combined dataframe."""
    if base_data_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            project_root / "MAIN_DATA" / "MST_DATA",
            project_root / "MAIN_DATA" / "MST_Data",
            Path(__file__).resolve().parent / "MST_Data",
        ]
        base_data_dir = next((path for path in candidates if path.exists()), candidates[1])

    conditions = ["item_only", "Both_item_task", "task_only"]
    frames: List[pd.DataFrame] = []

    for condition in conditions:
        condition_path = base_data_dir / condition
        if not condition_path.exists():
            print(f"Warning: missing condition folder: {condition_path}")
            continue

        frames.append(process_condition(condition_path))

    if not frames:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "participant_uid",
                "condition",
                "image_path",
                "trial_type",
                "user_response",
                "boundary_position",
                "RT",
                "encoding_RT",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    combined_df = main()
    output_dir = Path(__file__).resolve().parent

    output_path = output_dir / "compiled_preprocessed.csv"
    combined_df.to_csv(output_path, index=False)

    condition_output_names = {
        "Both_item_task": "both_item_task_preprocessed.csv",
        "item_only": "item_only_preprocessed.csv",
        "task_only": "task_only_preprocessed.csv",
    }
    for condition_name, filename in condition_output_names.items():
        condition_df = combined_df[combined_df["condition"] == condition_name].copy()
        condition_df.to_csv(output_dir / filename, index=False)

    print(f"Saved combined dataframe to: {output_path}")
    print("Saved condition dataframes to: both_item_task_preprocessed.csv, item_only_preprocessed.csv, task_only_preprocessed.csv")
    print(f"Combined dataframe shape: {combined_df.shape}")
    print(combined_df.head())
