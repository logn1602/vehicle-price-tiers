# Data

The raw dataset is **not committed** — it is ~1.35 GB and `.gitignore` excludes
`*.csv` and `*.parquet` from the repository. Download it yourself with the steps
below; the pipeline verifies the file is correct before doing anything with it.

## Source

**Kaggle:** [`austinreese/craigslist-carstrucks-data`](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

Used-vehicle listings scraped from Craigslist across all U.S. states. The scrape
dates from **2021**, which is why every age-derived feature uses `current_year = 2021`
rather than the current calendar year.

## Download (PowerShell)

Authenticate the Kaggle CLI once — place your `kaggle.json` API token at
`%USERPROFILE%\.kaggle\kaggle.json`, then:

```powershell
pip install kaggle
kaggle datasets download -d austinreese/craigslist-carstrucks-data -p data\raw
Expand-Archive data\raw\craigslist-carstrucks-data.zip -DestinationPath data\raw
Remove-Item data\raw\craigslist-carstrucks-data.zip
```

You should end up with `data/raw/vehicles.csv`.

## Expected shape

The ingest layer asserts these on load and **fails the run** if they do not hold,
rather than proceeding on a wrong or partially-downloaded file. This assertion is
the single check that would have caught the defect described in the engineering
notes of the top-level README.

| Property | Value |
|---|---|
| Rows | 426,880 |
| Columns | 26 |

Dataset versions on Kaggle have been revised over time. If the assertion fails on
a fresh download, you have a different revision than the one this project was
built against — check `results/v1_load_diagnosis.json` for the recorded shape and
file size of the reference copy.

## Layers

The pipeline writes three derived layers, all gitignored:

| Path | Contents |
|---|---|
| `data/bronze/` | Typed Parquet, one-to-one with the raw CSV. No rows dropped. |
| `data/silver/` | Cleaned and filtered, partitioned by `state`. Lineage-tracked. |
| `data/gold/` | Model-ready feature store. |

Every transition between layers appends a record to `results/lineage.json` with
rows in, rows out, and percentage dropped. Any stage that drops more than 25% of
its input fails the run unless whitelisted in `conf/config.yaml` with a reason.

## A note on `price`

`price` is the seller's **asking price**, not a transaction price. Nothing in this
dataset records what a vehicle actually sold for, or whether it sold at all. Every
result in this repository is a model of listing behaviour, not market value.
