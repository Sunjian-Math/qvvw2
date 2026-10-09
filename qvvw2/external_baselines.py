"""Load the optional Marginal-W2sq baseline from its authors' fixed Git revision.

No upstream modules are distributed with qvvw2. The first use downloads the
required files to a user-side cache. Installing qvvw2 itself needs no network.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile

WAVEFORM_OT_COMMIT = "b4d0b87130a5fef0621f0966994e94db8b18e71a"
WAVEFORM_OT_SOURCE_URL = "https://github.com/msambridge/waveform-ot"
WAVEFORM_OT_ARCHIVE_URL = (
    WAVEFORM_OT_SOURCE_URL + "/archive/" + WAVEFORM_OT_COMMIT + ".zip"
)
REQUIRED_MODULES = (
    "OTlib.py", "FingerprintLib.py", "ricker_util.py",
    "ricker_util_opt.py", "myGP.py",
)
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MODULE_BYTES = 8 * 1024 * 1024


def external_root() -> Path:
    override = os.getenv("QVVW2_EXTERNAL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / "qvvw2" / "external").resolve()
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return (Path(os.environ["LOCALAPPDATA"]) / "qvvw2" / "external").resolve()
    return (Path.home() / ".cache" / "qvvw2" / "external").resolve()


def _ready(repo: Path) -> bool:
    return all((repo / "libs" / name).is_file() for name in REQUIRED_MODULES)


def _source_files(archive: bytes) -> dict[str, bytes]:
    # Limit archive size and access only the explicitly named files: no arbitrary
    # ZIP extraction or user-supplied executable paths.
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("waveform-ot download exceeds the size limit")
    prefix = f"waveform-ot-{WAVEFORM_OT_COMMIT}/libs/"
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            result = {}
            for name in REQUIRED_MODULES:
                member = z.getinfo(prefix + name)
                if member.file_size > MAX_MODULE_BYTES:
                    raise RuntimeError("waveform-ot module exceeds the size limit: " + name)
                result[name] = z.read(member)
            return result
    except (zipfile.BadZipFile, KeyError, OSError, EOFError) as exc:
        raise RuntimeError("Cannot read required modules from waveform-ot source archive") from exc


def _get_author_archive() -> bytes:
    try:
        request = Request(WAVEFORM_OT_ARCHIVE_URL, headers={"User-Agent": "qvvw2/1.1.4"})
        with urlopen(request, timeout=45) as stream:
            payload = stream.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError(
            "Unable to download optional waveform-ot from its original repository. "
            "Check your connection or follow the offline instructions in README: "
            + WAVEFORM_OT_ARCHIVE_URL
        ) from exc
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("waveform-ot download exceeds the size limit")
    return payload


def ensure_waveform_ot() -> Path:
    """Download once if needed, then reuse the source held in the user cache."""
    root = external_root()
    repo = root / ("waveform-ot-" + WAVEFORM_OT_COMMIT)
    if repo.exists():
        if not _ready(repo):
            raise RuntimeError("waveform-ot cache is incomplete: " + str(repo))
        return repo
    if os.getenv("QVVW2_AUTO_FETCH", "1").lower() in ("0", "false", "no"):
        raise RuntimeError("waveform-ot has not been downloaded and auto-fetch is disabled")
    print("Downloading Sambridge et al. waveform-ot from the original repository ...", flush=True)
    modules = _source_files(_get_author_archive())
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".waveform-ot-", dir=root) as tmpdir:
        stage = Path(tmpdir) / "source"
        libs = stage / "libs"
        libs.mkdir(parents=True)
        (libs / "__init__.py").write_text("")
        for filename, content in modules.items():
            (libs / filename).write_bytes(content)
        if repo.exists():
            if not _ready(repo):
                raise RuntimeError("waveform-ot cache is incomplete: " + str(repo))
        else:
            try:
                stage.rename(repo)
            except OSError as exc:
                if not _ready(repo):
                    raise RuntimeError("Failed to install waveform-ot in the local cache") from exc
    return repo


def waveform_ot_commit() -> str:
    return WAVEFORM_OT_COMMIT
