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
monitor_app = typer.Typer(help="Stability monitoring and reject inference.", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")
mm_app = typer.Typer(
    help="Thin-file track on SYNTHETIC Ethiopian mobile-money borrowers.", no_args_is_help=True
)
app.add_typer(mm_app, name="mm")

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
MM_DIR = Path("data/mobile_money")  # synthetic
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
def data_download(
    force: Annotated[bool, typer.Option(help="Re-download.")] = False,
    rejected: Annotated[bool, typer.Option(help="Also the rejected file (255 MB).")] = False,
) -> None:
    """Download the Lending Club accepted-loans file from Kaggle (393 MB)."""
    from crisk.data.kaggle import ACCEPTED, REJECTED, download

    files = (ACCEPTED, REJECTED) if rejected else (ACCEPTED,)
    for path in download(RAW_DIR, files=files, force=force):
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
    save(res, REPORTS_DIR, MODELS_DIR)
    typer.echo(f"calibration chosen on validation (out-of-fold Brier): {res.chosen}")
    cal = res.calibration[res.calibration["split"] == "test"]
    cols = ["model", "version", "brier", "ece", "citl", "slope", "mean_pd", "default_rate"]
    typer.echo(cal[cols].round(4).to_string(index=False))
    typer.echo(f"\ncutoffs chosen on validation: {res.cutoffs}")
    cols = ["policy", "objective", "split", "approval_rate", "profit", "return_on_lent", "bad_rate"]
    typer.echo(res.policies[cols].round(4).to_string(index=False))


@app.command()
def explain(
    ablate: Annotated[
        bool, typer.Option(help="Refit the champion with addr_state toggled.")
    ] = True,
) -> None:
    """SHAP for the champion, reason codes for every decline, fairness, the addr_state call."""
    from crisk.explain.run import run, save

    res = run(
        PROCESSED_DIR / "model_table.parquet",
        PROCESSED_DIR / "outcomes.parquet",
        MODELS_DIR,
        ablate=ablate,
    )
    save(res, REPORTS_DIR)
    typer.echo(res.importance.head(12).round(4).to_string(index=False))
    typer.echo("\n" + res.reason_frequency.round(3).to_string(index=False))
    typer.echo("\n" + res.reason_counts.to_string(index=False))
    cols = ["policy", "attribute", "groups_used", "min_air", "min_air_group", "max_tpr_gap"]
    typer.echo("\n" + res.fairness_summary[cols].round(3).to_string(index=False))
    if ablate:
        typer.echo("\n" + res.state_ablation.round(4).to_string(index=False))
        typer.echo(f"\naddr_state decision (validation only): {res.state_decision}")


@data_app.command("audit")
def data_audit() -> None:
    """Print the leakage audit (every column that is not a model input, and why)."""
    from crisk.data.columns import audit_markdown

    typer.echo(audit_markdown())


@mm_app.command("generate")
def mm_generate(
    n: Annotated[int, typer.Option(help="Synthetic borrowers.")] = 20_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Generate SYNTHETIC borrowers and wallet history, then build their features."""
    from crisk.mobile_money.features import build
    from crisk.mobile_money.generate import generate

    book = generate(n=n, seed=seed)
    MM_DIR.mkdir(parents=True, exist_ok=True)
    book.borrowers.to_parquet(MM_DIR / "borrowers.parquet", index=False)
    book.transactions.to_parquet(MM_DIR / "transactions.parquet", index=False)
    features = build(book.transactions, book.borrowers)
    features.to_parquet(MM_DIR / "features.parquet", index=False)
    b = book.borrowers
    typer.echo(
        f"SYNTHETIC: {len(b):,} borrowers, {len(book.transactions):,} transactions, "
        f"default rate {b['default'].mean():.1%}, {len(features.columns) - 2} features -> {MM_DIR}"
    )


@mm_app.command("train")
def mm_train() -> None:
    """Scorecard, LightGBM, calibration, cutoff, reason codes and fairness (synthetic track)."""
    import pandas as pd

    from crisk.mobile_money.run import run, save

    res = run(
        pd.read_parquet(MM_DIR / "features.parquet"), pd.read_parquet(MM_DIR / "borrowers.parquet")
    )
    save(res, REPORTS_DIR, MODELS_DIR)
    cols = ["model", "split", "auc", "ks", "brier", "mean_pd", "default_rate"]
    typer.echo(res.metrics[cols].round(4).to_string(index=False))
    typer.echo(f"\nchampion (validation AUC): {res.champion}")
    typer.echo(res.lift.round(4).to_string(index=False))
    typer.echo("\n" + res.calibration.round(4).to_string(index=False))
    cols = ["policy", "split", "approval_rate", "profit", "return_on_lent", "bad_rate"]
    typer.echo("\n" + res.policies[cols].round(4).to_string(index=False))
    typer.echo("\n" + res.recovery.round(3).to_string(index=False))
    typer.echo("\n" + res.reason_frequency.round(3).to_string(index=False))
    typer.echo("\n" + res.reason_counts.to_string(index=False))
    cols = ["policy", "attribute", "groups_used", "min_air", "min_air_group", "max_tpr_gap"]
    typer.echo("\n" + res.fairness_summary[[*cols, "max_abs_citl"]].round(3).to_string(index=False))


@mm_app.command("score")
def mm_score(
    borrower_id: Annotated[list[int], typer.Argument(help="Synthetic borrower ids to score.")],
) -> None:
    """Score synthetic borrowers with the saved thin-file model: PD, decision, reasons."""
    import joblib
    import pandas as pd

    from crisk.mobile_money.run import MODEL_FILE, score

    bundle = joblib.load(MODELS_DIR / MODEL_FILE)
    features = pd.read_parquet(MM_DIR / "features.parquet")
    rows = features[features["borrower_id"].isin(borrower_id)]
    if rows.empty:
        raise typer.BadParameter(f"no synthetic borrower with id in {borrower_id}")
    typer.echo(score(bundle, rows.reset_index(drop=True)).round(4).to_string(index=False))


@data_app.command("rejected")
def data_rejected() -> None:
    """Stream the rejected-applications CSV into parquet (shared columns, split windows)."""
    from crisk.data.kaggle import REJECTED
    from crisk.monitor.rejected import rejected_to_parquet

    report = rejected_to_parquet(REJECTED.local_path(RAW_DIR), INTERIM_DIR / "rejected.parquet")
    typer.echo(f"rows read {report.rows_read:,}, kept {report.rows_written:,}: {report.by_split}")


@monitor_app.command("stability")
def monitor_stability() -> None:
    """PSI of scores and key features by issue quarter, and the characteristic report."""
    from crisk.monitor.run import save_stability, stability_run

    res = stability_run(PROCESSED_DIR / "model_table.parquet", MODELS_DIR)
    save_stability(res, REPORTS_DIR)
    typer.echo(res.summary.round(3).to_string(index=False))
    typer.echo("\n" + res.latest_csr.round(2).to_string(index=False))


@monitor_app.command("reject")
def monitor_reject() -> None:
    """Reject inference (fuzzy augmentation, parcelling) on the shared columns."""
    from crisk.monitor import reject

    split = reject.accepted_issue_split(RAW_PARQUET)
    res = reject.run(PROCESSED_DIR / "model_table.parquet", INTERIM_DIR / "rejected.parquet", split)
    reject.save(res, REPORTS_DIR)
    for table in (res.population, res.models, res.decisions, res.flips, res.extras["reject_psi"]):
        typer.echo(table.round(4).to_string(index=False) + "\n")


@app.command("export")
def export_models(
    track: Annotated[str, typer.Option(help="lending_club, mobile_money or all.")] = "all",
) -> None:
    """Export the trained champions to models/serving/ for the API (no pickles served)."""
    from crisk.serve import export as ex

    out = ex.SERVING_DIR
    if track in ("all", "lending_club"):
        path = ex.export_lending_club(
            MODELS_DIR, PROCESSED_DIR / "model_table.parquet", REPORTS_DIR, out
        )
        typer.echo(f"exported {path}")
    if track in ("all", "mobile_money"):
        path = ex.export_mobile_money(MODELS_DIR, MM_DIR / "features.parquet", REPORTS_DIR, out)
        typer.echo(f"exported {path}")


@app.command()
def bootstrap(
    mm_borrowers: Annotated[int, typer.Option(help="Synthetic borrowers if none exist.")] = 20_000,
) -> None:
    """One-shot setup for `docker compose up`: migrate, then make sure both tracks are exported.

    Idempotent; per track, the first that applies wins:
    1. models/serving/<track>.json exists: nothing to do.
    2. The trained model exists locally (models/*.joblib and its data): export it.
    3. Lending Club with KAGGLE_API_TOKEN: download, build, train, decide, export (~45 min).
    4. Otherwise: Lending Club gets a DEMO model on synthetic LC-shaped loans (the real data
       can't ship with the repo); the mobile-money track is generated, trained and exported
       (it is synthetic by design).
    """
    from alembic import command
    from alembic.config import Config

    from crisk.config import get_settings
    from crisk.serve import export as ex

    out = ex.SERVING_DIR
    typer.secho("1/3 migrate database", bold=True)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")

    typer.secho("2/3 Lending Club model", bold=True)
    table = PROCESSED_DIR / "model_table.parquet"
    real = [MODELS_DIR / "gbm_with_pricing.joblib", MODELS_DIR / "decision.joblib", table]
    token = get_settings().kaggle_api_token
    if (out / "lending_club.json").exists():
        typer.echo("already exported")
    elif all(p.exists() for p in real):
        typer.echo(f"exported {ex.export_lending_club(MODELS_DIR, table, REPORTS_DIR, out)}")
    elif token is not None and token.get_secret_value():
        typer.echo("Kaggle token set: running the real pipeline (download, build, train)")
        data_download(force=False, rejected=False)
        data_build()
        train_scorecard(mlflow=False)
        train_gbm(mlflow=False)
        decide()
        typer.echo(f"exported {ex.export_lending_club(MODELS_DIR, table, REPORTS_DIR, out)}")
    else:
        typer.echo("no Kaggle token: training the DEMO model on synthetic LC-shaped loans")
        typer.echo(f"exported {ex.export_demo_lending_club(out)}")

    typer.secho("3/3 mobile-money model (synthetic)", bold=True)
    if (out / "mobile_money.json").exists():
        typer.echo("already exported")
    else:
        if (
            not (MODELS_DIR / "mm_model.joblib").exists()
            or not (MM_DIR / "features.parquet").exists()
        ):
            mm_generate(n=mm_borrowers, seed=42)
            mm_train()
        path = ex.export_mobile_money(MODELS_DIR, MM_DIR / "features.parquet", REPORTS_DIR, out)
        typer.echo(f"exported {path}")
