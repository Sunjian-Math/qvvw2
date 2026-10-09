"""Tests for optional third-party source acquisition, without a live network."""
import io
import zipfile
import pytest
from qvvw2 import external_baselines as ext


def _archive():
    prefix = f"waveform-ot-{ext.WAVEFORM_OT_COMMIT}/libs/"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in ext.REQUIRED_MODULES:
            z.writestr(prefix + name, ("# source " + name + "\n").encode())
    return buf.getvalue()


def test_first_use_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("QVVW2_EXTERNAL_DIR", str(tmp_path))
    calls = []
    def download():
        calls.append(1)
        return _archive()
    monkeypatch.setattr(ext, "_get_author_archive", download)
    repo = ext.ensure_waveform_ot()
    assert all((repo / "libs" / name).is_file() for name in ext.REQUIRED_MODULES)
    assert ext.ensure_waveform_ot() == repo
    assert len(calls) == 1


def test_offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("QVVW2_EXTERNAL_DIR", str(tmp_path))
    monkeypatch.setenv("QVVW2_AUTO_FETCH", "0")
    with pytest.raises(RuntimeError, match="disabled"):
        ext.ensure_waveform_ot()


def test_incomplete_cache_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("QVVW2_EXTERNAL_DIR", str(tmp_path))
    (tmp_path / ("waveform-ot-" + ext.WAVEFORM_OT_COMMIT)).mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        ext.ensure_waveform_ot()


def test_invalid_archive_does_not_install(tmp_path, monkeypatch):
    monkeypatch.setenv("QVVW2_EXTERNAL_DIR", str(tmp_path))
    monkeypatch.setattr(ext, "_get_author_archive", lambda: b"not a ZIP")
    with pytest.raises(RuntimeError, match="archive"):
        ext.ensure_waveform_ot()
    assert not (tmp_path / ("waveform-ot-" + ext.WAVEFORM_OT_COMMIT)).exists()
