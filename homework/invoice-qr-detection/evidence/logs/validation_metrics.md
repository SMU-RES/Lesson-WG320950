# validation metrics

```text
Task 9 model validation evidence
timestamp_utc=2026-05-22T18:28:38.656749+00:00
command=.venv\Scripts\python.exe scripts\validate_model.py --weights runs\detect\train_qr\weights\best.pt --data dataset\data.yaml --split test --conf 0.25
exit_code=0
weights=runs/detect/train_qr/weights/best.pt
weights_bytes=5493265
final_weight=weights/best.pt
final_weight_bytes=5493265
data=dataset/data.yaml
split=test
conf=0.25
device=cpu
validation_run_dir=runs/detect/val_qr-2
metrics_source=metrics.box fields with results_dict fallback
precision=1.000000
recall=0.709570
mAP50=0.974510
mAP50-95=0.652415
mAP50_floor=0.80
mAP50_status=PASS
recall_floor=0.75
recall_status=MISS
negative_split=test
negative_image_count=10
false_positive_count=0
false_positive_floor=2
false_positive_status=PASS
limitation=none
```
