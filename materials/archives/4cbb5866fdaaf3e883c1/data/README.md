# Data

Use TRUDI as the primary source for demo photos:

https://github.com/egulsoylu/trudi

The visible repo README reports 733 images and 35034 annotated instances from ground and aerial perspectives. No clear LICENSE file or explicit license statement was visible in the inspected repository root/README, so cite the paper and use the data conservatively for coursework/demo use.

Keep the full dataset outside this repository or under `data/TRUDI/`, which is ignored by git. Copy a small selected subset into `data/trudi_samples/` with:

```powershell
uv run python scripts/prepare_trudi_samples.py --source C:\path\to\TRUDI --limit 20 --seed 42 --out data\trudi_samples
```
