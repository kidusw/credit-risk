from pathlib import Path

import typer

app = typer.Typer(help="Credit risk and loan default: PD, scorecards, expected loss, reason codes.", no_args_is_help=True)
data_app = typer.Typer(help="Download and build datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
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
