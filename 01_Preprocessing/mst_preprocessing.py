"""Preprocessing pipeline for modified Mnemonic Similarity Task (MST).

This version builds analysis-ready data directly from raw test logs in ``MST_Data``
and uses robust fallbacks from encoding logs when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_boundary_label(value: object) -> Optional[str]:
    """Normalize boundary labels to Pre/Mid/Post."""
    if pd.isna(value):
        return None

    label = str(value).strip().lower()
    mapping = {
        "pre": "Pre",
        "mid": "Mid",
        "post": "Post",
    }
    return mapping.get(label)


def label_boundary_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """Label boundaries from encoding order assuming 7-item event structure."""
    out = df.copy()
    out["event_number"] = out["trial_index"] // 7
    out["position_in_event"] = out["trial_index"] % 7
    out["boundary_position_from_task"] = np.where(
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
    """Create a canonical key so target/lure pairs map to one encoded stimulus.

    Examples:
    - Objects\123a.jpg -> Objects/123a.jpg
    - Objects\123b.jpg -> Objects/123a.jpg
    - Foils/scene7.jpg -> Foils/scene7.jpg (unchanged, no encoding match expected)
    """
    if pd.isna(path):
        return None

    normalized = str(path).strip().replace("\\", "/")
    if not normalized:
        return None

    lower = normalized.lower()
    if lower.endswith("a.jpg") or lower.endswith("b.jpg"):
        return normalized[:-5] + "a.jpg"
    return normalized


def _extract_response_key(df: pd.DataFrame) -> pd.Series:
    """Extract participant response key from available test response columns."""
    candidate_columns = ["trials.key_resp_3.keys", "key_resp_3.keys", "key_resp.keys"]
    for column in candidate_columns:
        if column in df.columns:
            return df[column]
    return pd.Series([np.nan] * len(df), index=df.index)


def _extract_test_rt(df: pd.DataFrame) -> pd.Series:
    """Extract recognition-test RT from available columns."""
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


def _extract_task_rt(row: pd.Series) -> float:
    """Compute encoding RT fallback for a task row."""
    rt9 = pd.to_numeric(row.get("key_resp_9.rt"), errors="coerce")
    if pd.notna(rt9):
        return float(rt9)

    rt8 = pd.to_numeric(row.get("key_resp_8.rt"), errors="coerce")
    if pd.notna(rt8):
        return float(3 + rt8)

    return np.nan


def process_participant(task_file: Path, test_file: Path, condition: str) -> pd.DataFrame:
    """Process one participant's task/test files into analysis-ready rows."""
    task_df = pd.read_csv(task_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)

    if "image_path" not in task_df.columns:
        raise ValueError(f"Missing 'image_path' in encoding file: {task_file}")
    if "image_path" not in test_df.columns:
        raise ValueError(f"Missing 'image_path' in test file: {test_file}")

    encoding_trials = task_df[task_df["image_path"].notna()].copy()
    encoding_trials = encoding_trials[encoding_trials["image_path"].astype(str).str.strip() != ""]

    encoding_trials = encoding_trials.reset_index(drop=True)
    encoding_trials["trial_index"] = np.arange(len(encoding_trials), dtype=int)
    encoding_trials = label_boundary_from_index(encoding_trials)
    encoding_trials["encoding_rt"] = encoding_trials.apply(_extract_task_rt, axis=1)

    boundary_lookup = encoding_trials[["image_path", "boundary_position_from_task", "encoding_rt"]].copy()
    boundary_lookup = boundary_lookup[boundary_lookup["image_path"].notna()]
    boundary_lookup["stimulus_key"] = boundary_lookup["image_path"].apply(canonical_stimulus_key)
    boundary_lookup = boundary_lookup.drop_duplicates(subset=["stimulus_key"], keep="first")

    test_trials = test_df[test_df["image_path"].notna()].copy()
    test_trials = test_trials[test_trials["image_path"].astype(str).str.strip() != ""]
    test_trials["trial_type"] = test_trials["image_path"].apply(classify_trial)
    test_trials["stimulus_key"] = test_trials["image_path"].apply(canonical_stimulus_key)
    test_trials["response_key"] = _extract_response_key(test_trials)
    test_trials["user_response"] = test_trials["response_key"].apply(_map_response_to_label)
    test_trials["RT"] = _extract_test_rt(test_trials)

    if "position_of_stimuli" in test_trials.columns:
        test_trials["boundary_position_from_test"] = test_trials["position_of_stimuli"].apply(normalize_boundary_label)
    else:
        test_trials["boundary_position_from_test"] = np.nan

    merged = test_trials.merge(
        boundary_lookup[["stimulus_key", "boundary_position_from_task", "encoding_rt"]],
        on="stimulus_key",
        how="left",
    )

    merged["boundary_position"] = merged["boundary_position_from_test"]
    needs_fallback = merged["boundary_position"].isna() & merged["trial_type"].isin(["Target", "Lure"])
    merged.loc[needs_fallback, "boundary_position"] = merged.loc[needs_fallback, "boundary_position_from_task"]

    missing_test_rt = merged["RT"].isna()
    merged.loc[missing_test_rt, "RT"] = merged.loc[missing_test_rt, "encoding_rt"]

    participant_id = _participant_id_from_name(task_file)
    if participant_id is None:
        participant_id = "unknown"
    participant_id = f"{condition}_{participant_id}"

    output = merged[["image_path", "trial_type", "user_response", "boundary_position", "RT"]].copy()
    output.insert(0, "condition", condition)
    output.insert(0, "participant_id", participant_id)
    return output


def _pair_files(csv_files: List[Path]) -> List[Tuple[str, Path, Path]]:
    """Pair task/test files by participant id; use latest file if duplicates exist."""
    task_files: Dict[str, List[Path]] = {}
    test_files: Dict[str, List[Path]] = {}

    for csv_file in csv_files:
        participant_id = _participant_id_from_name(csv_file)
        if participant_id is None:
            continue

        lower_name = csv_file.name.lower()
        if "_task_" in lower_name:
            task_files.setdefault(participant_id, []).append(csv_file)
        elif "_test_" in lower_name:
            test_files.setdefault(participant_id, []).append(csv_file)

    all_ids = sorted(set(task_files).union(test_files))
    pairs: List[Tuple[str, Path, Path]] = []

    for participant_id in all_ids:
        task_candidates = sorted(task_files.get(participant_id, []))
        test_candidates = sorted(test_files.get(participant_id, []))

        if not task_candidates or not test_candidates:
            print(
                f"Warning: pairing failed for participant {participant_id} "
                f"(task files: {len(task_candidates)}, test files: {len(test_candidates)})."
            )
            continue

        if len(task_candidates) > 1:
            print(f"Warning: multiple task files for participant {participant_id}; using latest by name.")
        if len(test_candidates) > 1:
            print(f"Warning: multiple test files for participant {participant_id}; using latest by name.")

        pairs.append((participant_id, task_candidates[-1], test_candidates[-1]))

    return pairs


def process_condition(
    condition_folder_path: Path,
    condition_label: str,
    supplemental_csv_files: Optional[List[Path]] = None,
) -> pd.DataFrame:
    """Process all paired participants under one condition folder.

    The function searches recursively for CSV files, identifies encoding/test files,
    pairs by 5-digit participant ID, warns on missing pairs, and concatenates outputs.
    """
    csv_files = sorted(condition_folder_path.rglob("*.csv"))
    if supplemental_csv_files:
        csv_files.extend(supplemental_csv_files)
    pairs = _pair_files(csv_files)
    participant_frames: List[pd.DataFrame] = []

    for participant_id, task_file, test_file in pairs:

        try:
            participant_frames.append(
                process_participant(task_file=task_file, test_file=test_file, condition=condition_label)
            )
        except Exception as exc:
            print(
                f"Warning: failed processing participant {participant_id} in "
                f"{condition_label}: {exc}"
            )

    if not participant_frames:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "condition",
                "image_path",
                "trial_type",
                "user_response",
                "boundary_position",
                "RT",
            ]
        )

    return pd.concat(participant_frames, ignore_index=True)


def main(base_data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run preprocessing across all conditions and return the combined dataframe."""
    if base_data_dir is None:
        base_data_dir = Path(__file__).resolve().parent.parent / "MST_Data"

    conditions = [
        ("item_only", "item_only", "item_only_data"),
        ("both_item_task", "Both_item_task", "both_data"),
        ("task_only", "task_only", "task_only_data"),
    ]
    frames: List[pd.DataFrame] = []

    project_root = Path(__file__).resolve().parent.parent
    supplemental_data_dir = project_root / "DATA"

    for label, folder_name, subfolder in conditions:
        condition_path = base_data_dir / folder_name / subfolder
        if not condition_path.exists():
            print(f"Warning: missing condition folder: {condition_path}")
            continue

        supplemental_csv_files: List[Path] = []
        if label == "both_item_task" and supplemental_data_dir.exists():
            # Include externally provided task logs (e.g., 00021/00023) when absent in raw folders.
            supplemental_csv_files = sorted(supplemental_data_dir.glob("*_MST_task_*.csv"))

        frames.append(
            process_condition(
                condition_path,
                condition_label=label,
                supplemental_csv_files=supplemental_csv_files,
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "condition",
                "image_path",
                "trial_type",
                "user_response",
                "boundary_position",
                "RT",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    combined_df = main()
    output_dir = Path(__file__).resolve().parent.parent / "MAIN_DATA" / "MST_Data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "compiled_preprocessed.csv"
    combined_df.to_csv(output_path, index=False)

    condition_output_names = {
        "both_item_task": "both_item_task_preprocessed.csv",
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
