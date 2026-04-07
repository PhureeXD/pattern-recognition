from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


JOIN_KEY = "TransactionID"
TARGET_COLUMN = "isFraud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Team 1 Member 1 and Member 2 parquet outputs into a combined modeling dataset."
    )
    parser.add_argument("--member1-train", type=Path, required=True, help="Path to train_member1_features.parquet")
    parser.add_argument("--member1-test", type=Path, required=True, help="Path to test_member1_features.parquet")
    parser.add_argument("--member2-train", type=Path, required=True, help="Path to train_member2_features.parquet")
    parser.add_argument("--member2-test", type=Path, required=True, help="Path to test_member2_features.parquet")
    parser.add_argument("--out-train", type=Path, required=True, help="Path for merged train parquet output")
    parser.add_argument("--out-test", type=Path, required=True, help="Path for merged test parquet output")
    parser.add_argument("--out-manifest", type=Path, required=True, help="Path for merged feature manifest JSON")
    parser.add_argument(
        "--overlap-strategy",
        choices=["error", "prefer-member2"],
        default="prefer-member2",
        help="How to handle overlapping feature columns between Member 1 and Member 2 outputs.",
    )
    return parser.parse_args()


def get_parquet_columns(path: Path, label: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return pq.ParquetFile(path).schema.names


def read_parquet(path: Path, label: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    print(f"Loading {label}...")
    df = pd.read_parquet(path, columns=columns)
    if JOIN_KEY not in df.columns:
        raise KeyError(f"{label} is missing required join key: {JOIN_KEY}")
    print(f"Loaded {label}: shape={df.shape}")
    return df


def validate_base_frames(train_columns: list[str], test_columns: list[str], label: str) -> None:
    if TARGET_COLUMN not in train_columns:
        raise ValueError(f"{label} train is missing {TARGET_COLUMN}")
    if TARGET_COLUMN in test_columns:
        raise ValueError(f"{label} test should not contain {TARGET_COLUMN}")

    if JOIN_KEY not in train_columns:
        raise ValueError(f"{label} train is missing {JOIN_KEY}")
    if JOIN_KEY not in test_columns:
        raise ValueError(f"{label} test is missing {JOIN_KEY}")

    train_features = sorted(set(train_columns) - {TARGET_COLUMN})
    test_features = sorted(test_columns)
    if train_features != test_features:
        train_only = sorted(set(train_features) - set(test_features))
        test_only = sorted(set(test_features) - set(train_features))
        raise ValueError(
            f"{label} train/test columns do not match. "
            f"Unique to train: {train_only}. Unique to test: {test_only}."
        )


def find_overlapping_feature_columns(member1_df: pd.DataFrame, member2_df: pd.DataFrame) -> list[str]:
    ignored = {JOIN_KEY, TARGET_COLUMN}
    return sorted((set(member1_df.columns) & set(member2_df.columns)) - ignored)


def find_overlapping_feature_columns_from_schemas(member1_columns: list[str], member2_columns: list[str]) -> list[str]:
    ignored = {JOIN_KEY, TARGET_COLUMN}
    return sorted((set(member1_columns) & set(member2_columns)) - ignored)


def build_member1_read_columns(member1_columns: list[str], overlap_columns: list[str]) -> list[str]:
    keep = [col for col in member1_columns if col not in overlap_columns]
    if JOIN_KEY not in keep:
        keep.append(JOIN_KEY)
    if TARGET_COLUMN in member1_columns and TARGET_COLUMN not in keep:
        keep.append(TARGET_COLUMN)
    return keep


def merge_pair(
    member1_df: pd.DataFrame,
    member2_df: pd.DataFrame,
    split_name: str,
    overlap_strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    overlap = find_overlapping_feature_columns(member1_df, member2_df)
    if overlap:
        if overlap_strategy == "error":
            raise ValueError(
                f"Cannot merge {split_name}: overlapping feature columns found between Member 1 and Member 2: {overlap}"
            )
        if overlap_strategy == "prefer-member2":
            member1_df = member1_df.drop(columns=overlap)
        else:
            raise ValueError(f"Unsupported overlap strategy: {overlap_strategy}")

    print(f"Merging {split_name}...")
    merged = member1_df.merge(member2_df, on=JOIN_KEY, how="inner", validate="one_to_one")

    if len(merged) != len(member1_df) or len(merged) != len(member2_df):
        raise ValueError(
            f"{split_name} row count changed during merge. "
            f"member1={len(member1_df)}, member2={len(member2_df)}, merged={len(merged)}"
        )

    print(f"Merged {split_name}: shape={merged.shape}")
    return merged, overlap


def normalize_merged_train_columns(train_df: pd.DataFrame) -> pd.DataFrame:
    if f"{TARGET_COLUMN}_x" in train_df.columns and f"{TARGET_COLUMN}_y" in train_df.columns:
        same_target = train_df[f"{TARGET_COLUMN}_x"].equals(train_df[f"{TARGET_COLUMN}_y"])
        if not same_target:
            raise ValueError("Member 1 and Member 2 train targets do not match.")
        train_df = train_df.drop(columns=[f"{TARGET_COLUMN}_y"]).rename(columns={f"{TARGET_COLUMN}_x": TARGET_COLUMN})
    elif TARGET_COLUMN not in train_df.columns:
        raise ValueError("Merged train output is missing target column after merge.")
    return train_df


def validate_merged_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if not train_df[JOIN_KEY].is_unique:
        raise ValueError(f"Merged train contains duplicate {JOIN_KEY} values")
    if not test_df[JOIN_KEY].is_unique:
        raise ValueError(f"Merged test contains duplicate {JOIN_KEY} values")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError("Merged train output is missing target column")
    if TARGET_COLUMN in test_df.columns:
        raise ValueError("Merged test output should not contain target column")

    train_features = sorted(set(train_df.columns) - {TARGET_COLUMN})
    test_features = sorted(test_df.columns)
    if train_features != test_features:
        train_only = sorted(set(train_features) - set(test_features))
        test_only = sorted(set(test_features) - set(train_features))
        raise ValueError(
            f"Merged train/test columns do not match. Unique to train: {train_only}. Unique to test: {test_only}."
        )


def build_manifest(
    member1_train: Path,
    member1_test: Path,
    member2_train: Path,
    member2_test: Path,
    out_train: Path,
    out_test: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dropped_overlap_columns: list[str],
    overlap_strategy: str,
    member1_columns: list[str],
    member2_columns: list[str],
) -> dict:
    merged_features = [col for col in train_df.columns if col != TARGET_COLUMN]

    return {
        "join_key": JOIN_KEY,
        "target_column": TARGET_COLUMN,
        "inputs": {
            "member1_train": str(member1_train),
            "member1_test": str(member1_test),
            "member2_train": str(member2_train),
            "member2_test": str(member2_test),
        },
        "outputs": {
            "merged_train": str(out_train),
            "merged_test": str(out_test),
        },
        "overlap_strategy": overlap_strategy,
        "dropped_member1_overlap_columns": dropped_overlap_columns,
        "member1_feature_count": len([col for col in member1_columns if col != TARGET_COLUMN]),
        "member2_feature_count": len([col for col in member2_columns if col != TARGET_COLUMN]),
        "merged_feature_count": len(merged_features),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "merged_columns": train_df.columns.tolist(),
    }


def main() -> None:
    args = parse_args()

    print("Reading parquet schemas...")
    member1_train_columns = get_parquet_columns(args.member1_train, "Member 1 train")
    member1_test_columns = get_parquet_columns(args.member1_test, "Member 1 test")
    member2_train_columns = get_parquet_columns(args.member2_train, "Member 2 train")
    member2_test_columns = get_parquet_columns(args.member2_test, "Member 2 test")

    validate_base_frames(member1_train_columns, member1_test_columns, "Member 1")
    validate_base_frames(member2_train_columns, member2_test_columns, "Member 2")

    train_overlap_columns = find_overlapping_feature_columns_from_schemas(member1_train_columns, member2_train_columns)
    test_overlap_columns = find_overlapping_feature_columns_from_schemas(member1_test_columns, member2_test_columns)

    if sorted(train_overlap_columns) != sorted(test_overlap_columns):
        raise ValueError(
            "Train/test overlap columns differ between Member 1 and Member 2 outputs. "
            f"Train overlap: {sorted(train_overlap_columns)}. Test overlap: {sorted(test_overlap_columns)}."
        )

    if train_overlap_columns:
        print(f"Detected overlapping columns: {len(train_overlap_columns)}")
        if args.overlap_strategy == "error":
            raise ValueError(
                f"Cannot merge: overlapping feature columns found between Member 1 and Member 2: {train_overlap_columns}"
            )
        print("Using overlap strategy: prefer-member2")

    member1_train_read_columns = build_member1_read_columns(member1_train_columns, train_overlap_columns)
    member1_test_read_columns = build_member1_read_columns(member1_test_columns, test_overlap_columns)

    member1_train = read_parquet(args.member1_train, "Member 1 train", columns=member1_train_read_columns)
    member1_test = read_parquet(args.member1_test, "Member 1 test", columns=member1_test_read_columns)
    member2_train = read_parquet(args.member2_train, "Member 2 train")
    member2_test = read_parquet(args.member2_test, "Member 2 test")

    if not member1_train[JOIN_KEY].is_unique:
        raise ValueError(f"Member 1 train contains duplicate {JOIN_KEY} values")
    if not member1_test[JOIN_KEY].is_unique:
        raise ValueError(f"Member 1 test contains duplicate {JOIN_KEY} values")
    if not member2_train[JOIN_KEY].is_unique:
        raise ValueError(f"Member 2 train contains duplicate {JOIN_KEY} values")
    if not member2_test[JOIN_KEY].is_unique:
        raise ValueError(f"Member 2 test contains duplicate {JOIN_KEY} values")

    merged_train, train_overlap = merge_pair(member1_train, member2_train, "train", args.overlap_strategy)
    merged_test, test_overlap = merge_pair(member1_test, member2_test, "test", args.overlap_strategy)
    merged_train = normalize_merged_train_columns(merged_train)

    validate_merged_frames(merged_train, merged_test)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    print("Writing merged train parquet...")
    merged_train.to_parquet(args.out_train, index=False)
    print("Writing merged test parquet...")
    merged_test.to_parquet(args.out_test, index=False)

    manifest = build_manifest(
        args.member1_train,
        args.member1_test,
        args.member2_train,
        args.member2_test,
        args.out_train,
        args.out_test,
        merged_train,
        merged_test,
        sorted(train_overlap),
        args.overlap_strategy,
        member1_train_columns,
        member2_train_columns,
    )
    print("Writing manifest...")
    with args.out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Merged outputs written:")
    print(f"- train: {args.out_train}")
    print(f"- test: {args.out_test}")
    print(f"- manifest: {args.out_manifest}")
    print(f"- merged train shape: {merged_train.shape}")
    print(f"- merged test shape: {merged_test.shape}")
    if train_overlap:
        print(f"- dropped overlapping Member 1 columns: {len(train_overlap)}")


if __name__ == "__main__":
    main()
