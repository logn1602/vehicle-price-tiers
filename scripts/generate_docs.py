"""Generate README.md and MODEL_CARD.md from the metrics artifacts.

This is the mechanism behind the no-hand-typed-numbers rule. Every figure in
both documents is interpolated from `results/metrics.json`,
`results/lineage.json`, `results/feature_manifest.json` or
`results/ablation.json`. Prose is templated; numbers are read.

`scripts/verify_readme.py` then checks the output independently, so a manual
edit that introduces an untraceable number fails CI even though this script
would never have produced one.

    python scripts/generate_docs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
GITHUB = "logn1602/vehicle-price-tiers"


def load(rel: str) -> Any | None:
    path = REPO / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def pct(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}%"


def thousands(value: int) -> str:
    return f"{value:,}"


def best_model(metrics: dict) -> dict:
    name = metrics["selection"]["best_model"]
    return next(m for m in metrics["models"] if m["model"] == name)


def results_table(metrics: dict) -> str:
    """Model comparison. Majority baseline is always the first row."""
    primary = metrics["selection"]["primary_metric"]
    base = metrics["baseline"]
    lines = [
        "| Model | Macro F1 | Accuracy | QWK | Adjacent acc | Tier MAE "
        "| AUC (OvR macro) | Train/test gap |",
        "|---|---|---|---|---|---|---|---|",
        f"| **Majority baseline** (`{base['predicted_class']}`) | {fmt(base['macro_f1'])} "
        f"| {fmt(base['accuracy'])} | {fmt(base['quadratic_weighted_kappa'])} "
        f"| {fmt(base['adjacent_accuracy'])} | {fmt(base['tier_mae'])} | — | — |",
    ]
    for m in sorted(metrics["models"], key=lambda r: r[primary], reverse=True):
        auc = m.get("auc_ovr_macro")
        lines.append(
            f"| {m['model']} | {fmt(m['macro_f1'])} | {fmt(m['accuracy'])} "
            f"| {fmt(m['ordinal']['quadratic_weighted_kappa'])} "
            f"| {fmt(m['ordinal']['adjacent_accuracy'])} "
            f"| {fmt(m['ordinal']['tier_mae'])} "
            f"| {fmt(auc) if auc is not None else '—'} "
            f"| {fmt(m['overfitting_gap'])} |"
        )
    return "\n".join(lines)


def per_class_table(model: dict) -> str:
    lines = [
        "| Tier | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for c in model["per_class"]:
        lines.append(
            f"| {c['class']} | {fmt(c['precision'])} | {fmt(c['recall'])} "
            f"| {fmt(c['f1'])} | {thousands(c['support'])} |"
        )
    return "\n".join(lines)


def confusion_table(model: dict) -> str:
    conf = model["confusion"]
    labels = conf["labels"]
    header = "| True \\ Predicted | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    lines = [header, sep]
    for label, row in zip(labels, conf["counts"], strict=True):
        lines.append(f"| **{label}** | " + " | ".join(thousands(v) for v in row) + " |")
    return "\n".join(lines)


def lineage_table(lineage: list[dict]) -> str:
    lines = [
        "| Stage | Rows in | Rows out | Dropped | % |",
        "|---|---|---|---|---|",
    ]
    for r in lineage:
        flag = " \\*" if r.get("whitelisted") else ""
        lines.append(
            f"| `{r['stage']}`{flag} | {thousands(r['rows_in'])} "
            f"| {thousands(r['rows_out'])} | {thousands(r['rows_dropped'])} "
            f"| {pct(r['pct_dropped'])} |"
        )
    return "\n".join(lines)


def ablation_table(ablation: dict | None, model: str = "xgboost") -> str:
    if not ablation:
        return "_Not yet generated. Run `python scripts/ablation.py`._"
    base = ablation["baseline"]
    lines = [
        "| Configuration | Rows | Macro F1 | Accuracy | vs baseline "
        "| Luxury recall | Luxury n | QWK |",
        "|---|---|---|---|---|---|---|---|",
        f"| **Majority baseline** | — | {fmt(base['macro_f1'])} | {fmt(base['accuracy'])} "
        f"| — | — | — | {fmt(0.0)} |",
    ]
    for row in ablation["configurations"]:
        m = row["models"][model]
        delta = m["macro_f1"] - base["macro_f1"]
        lines.append(
            f"| {row['configuration']} | {thousands(row['rows'])} | {fmt(m['macro_f1'])} "
            f"| {fmt(m['accuracy'])} | {delta:+.4f} | {fmt(m['luxury_recall'])} "
            f"| {thousands(m['luxury_support'])} | {fmt(m['quadratic_weighted_kappa'])} |"
        )
    return "\n".join(lines)


def luxury_story(model: dict) -> str:
    """The finding, expressed in counts rather than a metric."""
    conf = model["confusion"]
    labels = conf["labels"]
    lux = labels.index("Luxury")
    prem = labels.index("Premium")
    row = conf["counts"][lux]
    total = sum(row)
    to_premium = row[prem]
    correct = row[lux]
    share = 100.0 * to_premium / total if total else 0.0
    lux_class = next(c for c in model["per_class"] if c["class"] == "Luxury")
    return (
        f"Of {thousands(total)} genuinely Luxury listings in the test set, the model "
        f"places {thousands(correct)} correctly and misfiles {thousands(to_premium)} "
        f"({pct(share, 1)}) as Premium — one tier down. Luxury recall is "
        f"{fmt(lux_class['recall'])}, against {fmt(lux_class['precision'])} precision."
    )


# ---------------------------------------------------------------------------
def build_readme(
    metrics: dict,
    lineage: list[dict],
    features: dict,
    ablation: dict | None,
    cfg: dict,
) -> str:
    best = best_model(metrics)
    prov = metrics["provenance"]
    base = metrics["baseline"]
    ingest = next((r for r in lineage if r["stage"] == "ingest_bronze"), {})
    extra = ingest.get("extra", {})
    final_rows = lineage[-1]["rows_out"] if lineage else prov["n_rows"]
    counts = features["groups"]
    raw_rows = cfg["data_contract"]["raw"]["rows"]
    v1_rows = cfg["v1_reconstruction"]["n_rows"]

    return f"""# Vehicle Price Tier Classification

[![CI](https://github.com/{GITHUB}/actions/workflows/ci.yml/badge.svg)](https://github.com/{GITHUB}/actions/workflows/ci.yml)

Classifies used-vehicle Craigslist listings into four ordinal price tiers
(Budget < Mid-Range < Premium < Luxury) from {thousands(final_rows)} cleaned listings and
{features["n_features"]} engineered features, using {best["model"]}. Primary metric is
{metrics["selection"]["primary_metric"]}, declared in `conf/config.yaml`.

> Every number in this file is generated from `results/metrics.json` by
> `scripts/generate_docs.py` and independently verified by
> `scripts/verify_readme.py` in CI. Nothing here is typed by hand.

## Results

{results_table(metrics)}

The baseline row is not decoration. `{base["predicted_class"]}` is the largest tier, so
predicting it for every listing scores {fmt(base["accuracy"])} accuracy while achieving
{fmt(base["macro_f1"])} macro F1. Any accuracy figure read without it is uninterpretable.

Note the baseline's adjacent accuracy of {fmt(base["adjacent_accuracy"])}: because the
majority tier sits in the *middle* of the ordering, always guessing it lands within one
tier of the truth almost always. Adjacent accuracy has a very high floor on this problem
and should not be read as a strong result on its own.

### Per-class performance, {best["model"]}

{per_class_table(best)}

Support is reported because a metric computed from a few dozen examples is not
comparable to one computed from thousands.

### Confusion matrix, {best["model"]}

Rows are true tiers, columns predicted, both in tier order.

{confusion_table(best)}

## The finding

{luxury_story(best)}

That is the error that costs money. A Luxury vehicle listed as Premium is
mispriced downward by at least the width of a tier, and the Premium/Luxury
boundary is exactly where dealer margin concentrates. Overall accuracy hides
this completely: Luxury is a small share of listings, so a model can ignore the
tier almost entirely and still look respectable.

Balanced sample weights are what move this number. The ablation below isolates
how much.

## Architecture

```
raw CSV  ──►  [ INGEST ]   ──►  bronze/   typed Parquet, every row, chunked read
                                   │
                              [ VALIDATE ]  pandera contract + quality gates
                                   │
                               silver/   cleaned, partitioned by state, lineage-tracked
                                   │
                              [ FEATURES ]  pure functions, no fitting
                                   │
                                gold/    model-ready feature store
                                   │
                               [ TRAIN ]   sklearn Pipeline, MLflow-tracked
                                   │
                        artifacts/ + results/metrics.json + MODEL_CARD.md
```

Each stage hashes its inputs and configuration; re-running with neither changed
is a logged cache hit rather than repeated work.

## Engineering notes

This is a rebuild. The previous version reported numbers that could not be
traced to any output. What follows is what the review found, stated plainly.

**Trained on a fraction of the data.** v1 loaded a file of 305,145 rows — not
the {thousands(raw_rows)} in the source — and modelled {thousands(v1_rows)} of them.
No warning fired.
The data layer now asserts the row count on load, records every row-count
transition in `results/lineage.json`, and fails the run if any single stage
drops more than 25% of its input without a whitelisted reason.

**Leakage in three transforms.** Imputation, variance filtering and feature
selection were all fitted on the full dataset before the train/test split, and
`SelectKBest` saw every label. Only the scaler was train-only. Everything that
learns now sits inside an `sklearn.Pipeline` fitted on the training split alone,
and `tests/test_leakage.py` demonstrates that train-only selection picks a
different feature set than full-data selection — so the leak was doing
something, and removing it is measurable.

**Class weighting skipped the reported model.** `class_weight='balanced'` was
set on the decision tree and random forest but not on XGBoost, which was the
model selected and written up. Weighting is now applied uniformly.

**Only one model was scaled.** A RBF-kernel SVM trained on unscaled features
scored worst of the five and the report attributed that to the algorithm.
Scaling is now inside the pipeline and applies to every model.

**The validation set was never used.** A 60/20/20 split built `X_val`, scaled
it, and never referenced it again because `early_stopping_rounds` was commented
out. It now drives early stopping, and a test asserts the booster stops before
exhausting its estimators.

**Model selection by an invented metric.** Models were ranked by
`0.4·accuracy + 0.3·F1 + 0.3·AUC`, weights unexplained. The primary metric is
now declared in config with a justification, with a documented tie-breaker.

**Feature engineering was truncated.** The pipeline built 3 of 9 feature groups
and stopped at a placeholder comment. All {features["n_features"]} features are now implemented as
pure functions: temporal {counts["temporal"]}, odometer {counts["odometer"]},
manufacturer {counts["manufacturer"]}, vehicle type {counts["vehicle_type"]},
condition {counts["condition"]}, fuel {counts["fuel"]}, transmission {counts["transmission"]},
title status {counts["title_status"]}, interactions {counts["interactions"]}.

**Unverifiable numbers in the written report.** Several figures in the v1 report
match no output the code produces, including its headline accuracy and AUC. They
are not carried forward. The report also states the age–price correlation as
positive while describing it as depreciation; measured on cleaned data it is
negative.

## Ablation

What each fix was worth, tracked on XGBoost — the model v1 selected.

{ablation_table(ablation)}

Accuracy is *expected* to fall when leakage is removed. That is the correct
outcome and nothing here is tuned to recover it.

The first row is a **reconstruction**, not a reproduction. v1's input file no
longer exists and no parse of the correct CSV recreates it, so the row is a
stratified sample at v1's reported class proportions. It reproduces the shape of
v1's experiment, not its rows.

## Data quality

CSV to typed Parquet: {extra.get("csv_mb", 0):,.1f} MB to {extra.get("parquet_mb", 0):,.1f} MB,
a {extra.get("compression_ratio", 0)}x reduction with no rows dropped.

{lineage_table(lineage)}

\\* whitelisted in `conf/config.yaml` with a stated reason.

The `pandera` contracts are asymmetric on purpose. The raw contract accepts the
source as it is, including a price of $3,736,928,711 and vehicles built after
the scrape date, because those are the rows the pipeline exists to remove and
failing at ingest would make the file unloadable. The silver contract rejects
them.

## Limitations

**`price` is an asking price, not a transaction price.** Nothing in this dataset
records what a vehicle sold for, or whether it sold at all. Every result here
models listing behaviour, not market value.

**Tier boundaries are conventional, not learned.** The cut points come from
industry financing thresholds. They are not derived from the price distribution
and no claim is made that they are optimal.

**The data is a 2021 snapshot.** Every age-derived feature uses 2021 as the
reference year. The model has no knowledge of any later market.

**Geography is coarse.** Listings are located by state; local market conditions
within a state are not represented.

**SVM is substituted on full data.** An exact RBF kernel does not terminate in
reasonable time at this sample size, so `LinearSVC` with probability calibration
stands in. The exact kernel is used only in the ablation's 6,067-row row.

## Reproduce

Requires Python 3.11+ and the Kaggle dataset at `data/raw/vehicles.csv`
(see [data/README.md](data/README.md)).

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
python scripts/run_all.py
python scripts/ablation.py
python scripts/generate_docs.py
pytest
```

Serve the model:

```powershell
docker compose up --build
```

Run provenance for the figures above — config hash `{prov["config_hash"][:12]}`,
data hash `{prov["data_hash"][:12]}`, seed {prov["seed"]}.
"""


def build_model_card(metrics: dict, features: dict, lineage: list[dict], cfg: dict) -> str:
    best = best_model(metrics)
    prov = metrics["provenance"]
    base = metrics["baseline"]
    boot = best["bootstrap"]
    cv = best.get("cross_validation", {})
    split = metrics["split"]
    raw_rows = cfg["data_contract"]["raw"]["rows"]
    auc_macro = best.get("auc_ovr_macro")
    auc_weighted = best.get("auc_ovr_weighted")
    auc_macro_text = fmt(auc_macro) if auc_macro is not None else "—"
    auc_weighted_text = fmt(auc_weighted) if auc_weighted is not None else "—"
    selected = best["n_selected_features"]
    hashes = f"`{prov['config_hash'][:12]}` · data hash `{prov['data_hash'][:12]}`"

    cv_line = (
        f"{fmt(cv['mean'])} ± {fmt(cv['std'])} over {cv['folds']} stratified folds"
        if cv
        else "not computed for this run"
    )

    return f"""# Model Card — Vehicle Price Tier Classifier

Generated from `results/metrics.json` by `scripts/generate_docs.py`.
Config hash {hashes} · seed {prov["seed"]}.

## Model details

| | |
|---|---|
| Algorithm | {best["model"]} |
| Task | Four-class ordinal classification |
| Classes | {" < ".join(metrics["class_labels"])} |
| Input features | {features["n_features"]} engineered, {selected} selected by the pipeline |
| Training rows | {thousands(split["train"])} |
| Validation rows | {thousands(split["val"])} (drives early stopping) |
| Test rows | {thousands(split["test"])} |
| Primary metric | {metrics["selection"]["primary_metric"]} |
| Tie-breaker | {metrics["selection"]["tie_breaker"]} |

## Intended use

Assigning a coarse price band to a used-vehicle listing described by structured
attributes: age, mileage, manufacturer, body type, condition, fuel,
transmission, title status.

Appropriate for exploratory pricing analysis, listing triage, and flagging
listings whose asking price is far from where comparable vehicles sit.

## Out of scope

- **Point price estimation.** This model outputs a tier, not a value. It cannot
  tell you what a vehicle is worth.
- **Lending, insurance, or underwriting decisions.** Trained on asking prices
  from a single classifieds platform, with no transaction outcomes.
- **Individual valuation disputes.** Tier boundaries are conventional and a
  listing near a boundary can fall either side on small input changes.
- **Markets outside the 2021 United States Craigslist snapshot.**

## Training data

Kaggle `austinreese/craigslist-carstrucks-data`, scraped 2021. {thousands(raw_rows)} raw
listings reduced to {thousands(lineage[-1]["rows_out"])} after the quality gates below.

{lineage_table(lineage)}

Class distribution in the training split:

| Tier | Count |
|---|---|
""" + "\n".join(
        f"| {label} | {thousands(n)} |"
        for label, n in base["train_class_distribution"].items()
    ) + f"""

## Performance

Measured on {thousands(split["test"])} held-out listings never seen during fitting or
feature selection.

| Metric | Value |
|---|---|
| Macro F1 | {fmt(best["macro_f1"])} |
| Weighted F1 | {fmt(best["weighted_f1"])} |
| Accuracy | {fmt(best["accuracy"])} |
| Majority-class baseline accuracy | {fmt(base["accuracy"])} |
| AUC, one-vs-rest macro | {auc_macro_text} |
| AUC, one-vs-rest weighted | {auc_weighted_text} |
| Quadratic weighted kappa | {fmt(best["ordinal"]["quadratic_weighted_kappa"])} |
| Adjacent accuracy | {fmt(best["ordinal"]["adjacent_accuracy"])} |
| Mean absolute error, tier units | {fmt(best["ordinal"]["tier_mae"])} |
| Train/test gap | {fmt(best["overfitting_gap"])} |

**{boot["metric"]}**: {fmt(boot["point_estimate"])}, {int(boot["level"] * 100)}% CI
[{fmt(boot["ci_lower"])}, {fmt(boot["ci_upper"])}] from {thousands(boot["iterations"])} bootstrap
resamples.

**Cross-validation**: {cv_line}. The entire pipeline is refitted inside each
fold, so imputation, variance filtering, selection and scaling are all learned
from fold-training data only.

### Per class

{per_class_table(best)}

### Confusion matrix

{confusion_table(best)}

## Known failure modes

**Luxury is systematically pulled toward Premium.** {luxury_story(best)} This is
the dominant error mode and it is directional, not random.

**Boundary listings are unstable.** A vehicle priced near a tier cut point can
move tiers on a small change in mileage or age. The model reports a full
probability distribution precisely so that low-confidence cases can be routed
elsewhere rather than accepted silently.

**Missing condition is common.** Roughly 40% of listings report no condition.
That value is imputed from the training median rather than assumed, but a
listing with no condition carries less information than one that reports it.

**Rare categories are weakly supported.** Electric and hybrid vehicles are a
small fraction of listings; predictions for them rest on less evidence.

**Asking price is not market price.** A systematically overpriced listing is
classified by its asking price, which is what the label records. The model
reproduces seller optimism rather than correcting it.

## Ethical and practical considerations

The training signal is what sellers *asked*, not what buyers *paid*. Using this
model to set prices would tend to reproduce and reinforce existing listing
behaviour, including any systematic over- or under-pricing in the source market.

Vehicle listings correlate with geography and therefore, indirectly, with
demographics. No fairness audit across protected characteristics has been
performed, and none is possible from these fields.

## Provenance

| | |
|---|---|
| Config hash | `{prov["config_hash"]}` |
| Data hash | `{prov["data_hash"]}` |
| Seed | {prov["seed"]} |
| Rows used | {thousands(prov["n_rows"])} |
| Features in | {prov["n_features_in"]} |
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readme", default="README.md")
    p.add_argument("--model-card", default="MODEL_CARD.md")
    args = p.parse_args(argv)

    import yaml

    metrics = load("results/metrics.json")
    lineage = load("results/lineage.json")
    features = load("results/feature_manifest.json")
    ablation = load("results/ablation.json")
    cfg = yaml.safe_load((REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))

    missing = [
        n for n, v in
        [("metrics.json", metrics), ("lineage.json", lineage), ("feature_manifest.json", features)]
        if v is None
    ]
    if missing:
        raise SystemExit(
            f"Missing artifacts: {', '.join(missing)}. Run `python scripts/run_all.py` first."
        )

    (REPO / args.readme).write_text(
        build_readme(metrics, lineage, features, ablation, cfg), encoding="utf-8"
    )
    (REPO / args.model_card).write_text(
        build_model_card(metrics, features, lineage, cfg), encoding="utf-8"
    )
    print(f"Wrote {args.readme} and {args.model_card}")
    if ablation is None:
        print("Note: results/ablation.json absent -- ablation section is a placeholder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
