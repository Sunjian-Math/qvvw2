from __future__ import annotations

from pathlib import Path
import os
import shutil
import time
import urllib.request
import zipfile


WAVEFORM_OT_COMMIT = (
    "b4d0b87130a5fef0621f0966994e94db8b18e71a"
)

WAVEFORM_OT_ARCHIVE = (
    "https://codeload.github.com/msambridge/"
    f"waveform-ot/zip/{WAVEFORM_OT_COMMIT}"
)

REVISION_MARKER = ".qvvw2_waveform_ot_revision"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def external_root() -> Path:
    env = os.environ.get("QVVW2_EXTERNAL_DIR")

    if env:
        return Path(env).expanduser().resolve()

    return (project_root() / "external").resolve()


def _valid_reference(repo: Path) -> bool:
    if not repo.exists():
        return False

    marker = repo / REVISION_MARKER

    if not marker.exists():
        return False

    if marker.read_text(encoding="utf-8").strip() != WAVEFORM_OT_COMMIT:
        return False

    required = [
        repo / "libs" / "OTlib.py",
        repo / "libs" / "FingerprintLib.py",
        repo / "libs" / "ricker_util.py",
    ]

    return all(p.exists() for p in required)


def _download_reference(repo: Path, retries: int = 3) -> None:
    root = repo.parent
    root.mkdir(parents=True, exist_ok=True)

    archive = root / "waveform-ot.download.zip"
    unpack = root / "waveform-ot.unpack"

    last_error = None

    for attempt in range(1, retries + 1):

        shutil.rmtree(unpack, ignore_errors=True)

        if archive.exists():
            archive.unlink()

        try:
            print(
                f"[qvvw2] Downloading verified Marginal-W2sq "
                f"reference snapshot ({attempt}/{retries})...",
                flush=True,
            )

            request = urllib.request.Request(
                WAVEFORM_OT_ARCHIVE,
                headers={
                    "User-Agent": "qvvw2-reproducibility"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response, open(archive, "wb") as output:

                shutil.copyfileobj(
                    response,
                    output,
                )

            unpack.mkdir(
                parents=True,
                exist_ok=True,
            )

            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(unpack)

            folders = [
                p for p in unpack.iterdir()
                if p.is_dir()
            ]

            if len(folders) != 1:
                raise RuntimeError(
                    "Unexpected waveform-ot archive structure."
                )

            extracted = folders[0]

            required = [
                extracted / "libs" / "OTlib.py",
                extracted / "libs" / "FingerprintLib.py",
                extracted / "libs" / "ricker_util.py",
            ]

            if not all(p.exists() for p in required):
                raise RuntimeError(
                    "Required waveform-ot files are missing."
                )

            (
                extracted / REVISION_MARKER
            ).write_text(
                WAVEFORM_OT_COMMIT + "\n",
                encoding="utf-8",
            )

            if repo.exists():
                shutil.rmtree(repo)

            shutil.move(
                str(extracted),
                str(repo),
            )

            shutil.rmtree(
                unpack,
                ignore_errors=True,
            )

            if archive.exists():
                archive.unlink()

            print(
                "[qvvw2] waveform-ot reference snapshot ready.",
                flush=True,
            )

            return

        except Exception as exc:
            last_error = exc

            shutil.rmtree(
                unpack,
                ignore_errors=True,
            )

            if archive.exists():
                archive.unlink()

            if attempt < retries:
                print(
                    "[qvvw2] Download interrupted; retrying...",
                    flush=True,
                )
                time.sleep(2)

    raise RuntimeError(
        "Could not obtain the waveform-ot reference snapshot.\n"
        "Check the network connection and rerun the notebook.\n"
        "No experiment configuration has been modified.\n"
        f"Original error: {last_error}"
    )


def ensure_waveform_ot() -> Path:
    repo = external_root() / "waveform-ot"

    if _valid_reference(repo):
        return repo

    if repo.exists():
        shutil.rmtree(repo)

    _download_reference(repo)

    if not _valid_reference(repo):
        raise RuntimeError(
            "waveform-ot verification failed after download."
        )

    return repo


def waveform_ot_commit() -> str:
    return WAVEFORM_OT_COMMIT
