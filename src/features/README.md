# Feature Pipelines

## Member 1 + Member 2 Merge

Use `merge_member1_member2.py` to combine the parquet outputs from the Team 1 Member 1 and Member 2 notebooks into a single modeling dataset.

By default, the script uses `--overlap-strategy prefer-member2`, which drops overlapping columns from Member 1 before merging. This matches the current notebook outputs, where Member 1 still contains some original transaction columns that Member 2 also exports.

Example:

```powershell
python src/features/merge_member1_member2.py `
  --member1-train "C:\path\to\train_member1_features.parquet" `
  --member1-test "C:\path\to\test_member1_features.parquet" `
  --member2-train "C:\path\to\train_member2_features.parquet" `
  --member2-test "C:\path\to\test_member2_features.parquet" `
  --out-train "data\processed\train_member1_member2_merged.parquet" `
  --out-test "data\processed\test_member1_member2_merged.parquet" `
  --out-manifest "data\processed\member1_member2_merge_manifest.json"
```

The script validates:

- `TransactionID` exists and is unique
- train/test schemas match for each member output
- overlapping columns can be rejected or dropped from Member 1, depending on `--overlap-strategy`
- merged train/test schemas still match except for `isFraud`
