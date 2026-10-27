"""Download the Lending Club files from the Kaggle mirror.

Mirror: https://www.kaggle.com/datasets/wordsforthewise/lending-club (listed as CC0-1.0 on
Kaggle, but the data originates from LendingClub's public downloads, which had their own
terms). Used for research and demonstration only; the raw files are never redistributed.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from crisk.config import get_settings

SLUG = "wordsforthewise/lending-club"


@dataclass(frozen=True)
class KaggleFile:
    filename: str
    size_mb: int

    def local_path(self, raw_dir: Path) -> Path:
        return raw_dir / "kaggle" / SLUG.replace("/", "__") / self.filename


ACCEPTED = KaggleFile("accepted_2007_to_2018Q4.csv.gz", 393)
REJECTED = KaggleFile("rejected_2007_to_2018Q4.csv.gz", 255)  # Phase 7: reject inference


class KaggleDownloadError(RuntimeError):
    pass


def download(raw_dir: Path, files: tuple[KaggleFile, ...] = (ACCEPTED,), force: bool = False):
    """Download `files` into `raw_dir/kaggle/<owner>__<name>/` unless already there."""
    todo = [f for f in files if force or not f.local_path(raw_dir).exists()]
    if not todo:
        return [f.local_path(raw_dir) for f in files]
    token = get_settings().kaggle_api_token
    if token is None:
        raise KaggleDownloadError("KAGGLE_API_TOKEN is not set (add it to .env)")
    os.environ["KAGGLE_API_TOKEN"] = token.get_secret_value()
    from kaggle.api.kaggle_api_extended import KaggleApi  # authenticates on import

    api = KaggleApi()
    api.authenticate()
    for f in todo:
        dest = f.local_path(raw_dir).parent
        dest.mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_file(SLUG, f.filename, path=dest, force=force, quiet=False)
        except Exception as exc:
            raise KaggleDownloadError(f"could not download {f.filename} ({exc})") from exc
    return [f.local_path(raw_dir) for f in files]
