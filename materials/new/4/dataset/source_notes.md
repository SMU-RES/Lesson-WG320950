# Dataset source and license policy

This project uses a conservative manifest-first workflow.

## Inclusion rule

A sample source may move toward `pending_review` or later inclusion only when all of the following are true.

1. The exact source URL is recorded.
2. The license or usage terms are explicit enough for local dataset preparation and model training.
3. The images do not expose sensitive real invoice information beyond what is already publicly and safely released.
4. The source fits project scope, which is QR detection in invoice or receipt context, not OCR or content decoding.

## Exclusion rule

Keep a source as `exclude` or `reference_only` when any of the following applies.

1. License is blank, `unknown`, `to_verify`, challenge-only, or otherwise unclear.
2. The source is generic QR imagery with no invoice or receipt context and no clear role yet.
3. The source may contain real financial or personal data that has not been reviewed for privacy risk.
4. The source is only mentioned in planning notes and has not been downloaded or inspected.

## Status meanings

- `reference_only`: candidate source retained in the manifest for later human or scripted verification. Not training-ready.
- `exclude`: not allowed into `data_working/` or `dataset/` unless the blocking license or privacy issue is resolved.
- `pending_review`: locally controlled source path that can become usable after the future task creates actual files and records per-sample provenance.
- `include`: reserved for future rows that point to actual local samples with explicit acceptable license terms. No row should use this status in task 2 unless the file is present and the license is clear.

## Practical guidance for later tasks

- Do not copy any real-source image into `dataset/images/*` until its manifest row has an explicit acceptable license.
- Treat synthetic invoice QR generation as the safest scaling path.
- Keep real and synthetic origins explicit in the manifest.
- Prevent split leakage by grouping synthetic variants from the same template family into one split only.
- If a candidate source is only useful for hard negatives, say that in `notes` before any sample is copied.
