"""Create the minimal source-and-panel archive needed by the cloud runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile


TICKERS = ("AAPL", "IWM", "NVDA", "QQQ", "SPY", "TLT", "XLE", "XLF", "XLK", "XLV")


def _required_paths(package_root: Path) -> list[Path]:
    panel_root = package_root / "Data" / "NEW_PANEL_DECISION_V2"
    paths = sorted(package_root.glob("*.py"))
    paths.extend(
        [
            package_root / "requirements-cloud.txt",
            package_root / "cloud_setup.sh",
            package_root / "CLOUD_RUN.md",
            package_root / "notebooks" / "protohedge-definitive-empirical-rerun.ipynb",
            panel_root / "panel_manifest.csv",
            panel_root / "panel_summary.json",
            panel_root / "_episode_diagnostics.csv",
        ]
    )
    paths.extend(sorted((package_root / "tests").glob("test_*.py")))
    for ticker in TICKERS:
        paths.extend(
            [
                panel_root / "episodes" / f"{ticker}_training_paths.npy",
                panel_root / "episodes" / f"{ticker}_chronological_splits.npz",
                panel_root / "episode_metadata" / f"{ticker}_episode_metadata.csv",
                panel_root / "contract_paths" / f"{ticker}_contract_paths.csv",
            ]
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot build cloud bundle; missing: " + ", ".join(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dist" / "protohedge-cloud.tar.gz",
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = _required_paths(package_root)
    archive_root = package_root.name
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            relative = path.relative_to(package_root)
            archive.add(path, arcname=str(Path(archive_root) / relative), recursive=False)
    print(f"Created {output} with {len(paths)} files ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
