# Public real-context data limitation note

Task 4 intentionally did not include public real invoice, receipt, QR-bill, or QR-context images as training-ready samples because the project policy requires exact source URL, explicit acceptable license/status, and low privacy risk before any real public sample can be marked `include`.

## Result

- `data_raw/` exists for future raw downloads.
- `data/working/public/` exists for future selected public working samples.
- Current safe public working image count: **0**.
- Current manifest rows marked `include` for real public samples: **0**.
- The target of at least 50 real-context public images was **not met**.

## Reasoning

The existing manifest remains conservative because the candidate sources were not all quickly and safely reducible to per-sample training-ready files during Task 4:

- SWICO QR-Bill: useful context/specification source, but not a clean open image dataset with clear redistribution/training terms.
- CIVQA: public mirrors exist, but license metadata is inconsistent and images may contain real invoice/business information.
- FATURA: Zenodo record indicates CC BY 4.0 and synthetic invoice images, but the archive is large and was not downloaded or inspected per-sample in this task.
- SROIE: official source is a challenge download with registration/terms; mirror licensing does not remove the need to respect official challenge restrictions and privacy review.
- Voxel51 consolidated receipt dataset: Hugging Face card indicates CC BY 4.0, but the dataset aggregates real receipt imagery and still needs per-sample privacy/provenance review before inclusion.
- Generic QR repositories may have clear software/data licenses, but they are not invoice/receipt context substitutes and should not be used to satisfy the 50 real-context target.

## Forward path

Task 5 synthetic invoice QR generation is the safe scaling path for this project. Future work may add real public samples only after downloading a bounded subset, inspecting privacy suitability, recording exact per-sample provenance, and updating `data/dataset_manifest.csv` with non-blank acceptable licenses and source URLs before setting `include_status=include`.
