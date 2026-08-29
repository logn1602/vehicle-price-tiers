# Forensics

One-off diagnostics from the audit of the previous version. They are kept
because the engineering notes in the top-level README make claims about what
went wrong, and a claim without its evidence is the thing this rebuild exists to
stop.

Neither script is part of the pipeline. Both read `data/raw/vehicles.csv`
directly and write to `results/`.

| Script | Answers |
|---|---|
| `diagnose_v1_load.py` | Is the CSV the genuine 426,880-row file? Do the stated data facts hold? Can any parse configuration reproduce the 305,145 rows v1 reported loading? |
| `diagnose_v1_rowloss.py` | Why do some of those facts miss? What actually collapsed v1's frame to 6,067 rows? |

Run from the repository root:

```powershell
python scripts/forensics/diagnose_v1_load.py
python scripts/forensics/diagnose_v1_rowloss.py
```

## What they established

**The file is genuine.** Nine facts match exactly, including `year` nulls at
1,205, `odometer` nulls at 4,400, and a maximum price of $3,736,928,711 to the
dollar.

**Seven other facts appeared to miss, and did not.** They were measured against
the 421,344-row frame that remains after dropping null `year`/`odometer`, not
against the raw 426,880. Recomputed on that denominator, all eight match to the
digit. The two denominators are now separate contracts in `conf/config.yaml`.

**v1 read a different file.** No parse configuration reproduces 305,145 rows;
the CSV contains zero malformed lines. Its price column must have been largely
unusable, since v1's own filter retains 92.3% of rows here and retained 2% there.

**The duplicate check v1 ran was meaningless.** It included `id`, `url` and
`image_url`, so it returned zero by construction. Excluding those, the real
count is 20.

Full output is in `results/v1_load_diagnosis.json` and
`results/v1_rowloss_diagnosis.json`.
