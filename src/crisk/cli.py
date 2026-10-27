from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Credit risk and loan default: PD, scorecards, expected loss, reason codes.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Download and build datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")
train_app = typer.Typer(help="Train and evaluate PD models.", no_args_is_help=True)
app.add_typer(train_app, name="train")

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
RAW_PARQUET = INTERIM_DIR / "accepted.parquet"
ROOT = Path(__file__).resolve().parents[2]


@app.callback()
def main() -> None:
    """crisk command-line interface."""


@app.command()
def migrate() -> None:
    """Apply database migrations (alembic upgrade head)."""
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@data_app.command("download")
def data_download(force: Annotated[bool, typer.Option(help="Re-download.")] = False) -> None:
    """Download the Lending Club accepted-loans file from Kaggle (393 MB)."""
    from crisk.data.kaggle import download

    for path in download(RAW_DIR, force=force):
        typer.echo(f"{path} ({path.stat().st_size / 2**20:.0f} MB)")


@data_app.command("ingest")
def data_ingest(
    chunksize: Annotated[int, typer.Option(help="CSV rows per chunk.")] = 200_000,
) -> None:
    """Stream the CSV into a column-pruned parquet file (data/interim)."""
    from crisk.data.ingest import csv_to_parquet
    from crisk.data.kaggle import ACCEPTED

    report = csv_to_parquet(ACCEPTED.local_path(RAW_DIR), RAW_PARQUET, chunksize=chunksize)
    typer.echo(
        f"rows={report.rows_written:,} (footer rows dropped: {report.footer_rows}) "
        f"chunks={report.chunks} peak_rss={report.peak_rss_mb:,.0f} MB "
        f"parquet={report.parquet_mb:,.0f} MB"
    )


@data_app.command("build")
def data_build() -> None:
    """Target, matured-vintage filter, time split, schema check -> data/processed."""
    import pandas as pd

    from crisk.data.build import build, write_reports
    from crisk.data.schema import validate

    if not RAW_PARQUET.exists():
        data_ingest()
    report = build(RAW_PARQUET, PROCESSED_DIR)
    validate(pd.read_parquet(PROCESSED_DIR / "model_table.parquet"))
    write_reports(report, REPORTS_DIR)
    for f in report.filters:
        typer.echo(f"{f['step']:<36} rows={f['rows']:>9,} removed={f['removed']:>9,}")
    typer.echo(report.splits.to_string(index=False))
    typer.echo(f"model table in memory: {report.model_table_mb:,.0f} MB; reports in {REPORTS_DIR}/")


@data_app.command("load")
def data_load() -> None:
    """Load matured loans and the vintage table into Postgres (idempotent)."""
    import pandas as pd

    from crisk.db import get_engine
    from crisk.db.loader import load, loan_rows

    loans = loan_rows(
        pd.read_parquet(PROCESSED_DIR / "model_table.parquet"),
        pd.read_parquet(PROCESSED_DIR / "outcomes.parquet"),
    )
    vintages = pd.read_csv(REPORTS_DIR / "vintages.csv")
    report = load(get_engine(), loans, vintages)
    typer.echo(f"loans={report.loans:,} vintages={report.vintages}")


@train_app.command("scorecard")
def train_scorecard(
    mlflow: Annotated[bool, typer.Option(help="Log runs to MLflow.")] = True,
) -> None:
    """WoE + logistic scorecard, with and without Lending Club's pricing columns."""
    from crisk.models.train import log_mlflow, save, train_variants

    results, baseline, champion = train_variants(PROCESSED_DIR / "model_table.parquet")
    save(results, baseline, champion, MODELS_DIR, REPORTS_DIR)
    if mlflow:
        log_mlflow(results, baseline, champion, REPORTS_DIR)
    for r in [*results, baseline]:
        n_features = len(getattr(r.scorecard, "selected", ["sub_grade"]))
        typer.echo(f"\n{r.name}: {n_features} features")
        cols = ["n", "default_rate", "auc", "gini", "ks", "pr_auc", "brier"]
        typer.echo(r.metrics[cols].round(4).to_string())
    typer.echo(f"\nchampion (validation AUC): {champion}")


@train_app.command("gbm")
def train_gbm(
    configs: Annotated[int, typer.Option(help="Random-search configurations per variant.")] = 12,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples for the AUC lift.")] = 200,
    mlflow: Annotated[bool, typer.Option(help="Log runs to MLflow.")] = True,
) -> None:
    """Monotonic LightGBM challenger, tuned on validation, compared with the scorecards."""
    from crisk.models import train_gbm as tg

    gbms, tuning, results, lifts, champion = tg.run(
        PROCESSED_DIR / "model_table.parquet", MODELS_DIR, n_configs=configs, n_boot=n_boot
    )
    tg.save(gbms, tuning, results, lifts, champion, MODELS_DIR, REPORTS_DIR)
    if mlflow:
        tg.log_mlflow(gbms, tuning, results, lifts, champion, REPORTS_DIR)
    rows = [
        {"model": name, "split": split, **r.metrics.loc[split, ["auc", "gini", "ks", "brier"]]}
        for name, r in results.items()
        for split in ("valid", "test")
    ]
    import pandas as pd

    typer.echo(pd.DataFrame(rows).round(4).to_string(index=False))
    cols = ["challenger", "incumbent", "split", "lift", "ci_low", "ci_high"]
    typer.echo("\n" + lifts[cols].round(4).to_string(index=False))
    typer.echo(f"\nchampion (validation AUC): {champion}")


@app.command()
def decide() -> None:
    """Calibrate PDs, expected vs realised loss, and the profit-maximising approval cutoff."""
    from crisk.decision.run import run, save

    res = run(PROCESSED_DIR / "model_table.parquet", PROCESSED_DIR / "outcomes.parquet", MODELS_DIR)
    save(res, REPORTS_DIR)
    typer.echo(f"calibration chosen on validation (out-of-fold Brier): {res.chosen}")
    cal = res.calibration[res.calibration["split"] == "test"]
    cols = ["model", "version", "brier", "ece", "citl", "slope", "mean_pd", "default_rate"]
    typer.echo(cal[cols].round(4).to_string(index=False))
    typer.echo(f"\ncutoffs chosen on validation: {res.cutoffs}")
    cols = ["policy", "objective", "split", "approval_rate", "profit", "return_on_lent", "bad_rate"]
    typer.echo(res.policies[cols].round(4).to_string(index=False))


@data_app.command("audit")
def data_audit() -> None:
    """Print the leakage audit (every column that is not a model input, and why)."""
    from crisk.data.columns import audit_markdown

    typer.echo(audit_markdown())
