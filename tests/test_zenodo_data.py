import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heterosc import zenodo
from heterosc.data import keep_complete_cell_lines, load_cell_line_embeddings


def _make_zip(tmp, name, with_root):
    z = tmp / f"{name}.zip"
    with zipfile.ZipFile(z, "w") as f:
        prefix = f"{name}/" if with_root else ""
        f.writestr(prefix + "raw/a.txt", "x")
        f.writestr(prefix + "summary.csv", "a,b\n1,2\n")
    return z


@pytest.mark.parametrize("with_root", [True, False])
def test_extract_zip_with_or_without_top_level_folder(tmp_path, with_root):
    z = _make_zip(tmp_path, "annotation", with_root)
    out = zenodo.extract_archive(z, tmp_path / "data")
    assert out == tmp_path / "data" / "annotation"
    assert (out / "raw" / "a.txt").exists() and (out / "summary.csv").exists()


def test_download_verifies_md5_and_rejects_corruption(tmp_path):
    src = tmp_path / "f.bin"; src.write_bytes(b"hello zenodo")
    good = zenodo.md5sum(src)
    p = zenodo.download_file(src.as_uri(), tmp_path / "out" / "f.bin", good, verbose=False)
    assert p.read_bytes() == b"hello zenodo"
    with pytest.raises(IOError, match="MD5 mismatch"):
        zenodo.download_file(src.as_uri(), tmp_path / "out2" / "f.bin", "0" * 32, verbose=False)
    assert not (tmp_path / "out2" / "f.bin").exists()                          # corrupt download is not kept


def test_record_constants():
    assert zenodo.ZENODO_DOI == "10.5281/zenodo.23153537" and len(zenodo.FILES) == 3
    assert all(len(v) == 32 for v in zenodo.FILES.values())


def test_unknown_part_and_missing_rar_tool(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        zenodo.download_data(tmp_path, parts=["nope"])
    monkeypatch.setattr(zenodo.shutil, "which", lambda _: None)
    r = tmp_path / "x.rar"; r.write_bytes(b"Rar!")
    with pytest.raises((RuntimeError, ImportError)):
        zenodo.extract_archive(r, tmp_path / "d")


def test_cell_line_embeddings_roundtrip(tmp_path):
    from heterosc.extraction.common import save_embeddings
    for e, d in (("a", 4), ("b", 6)):
        save_embeddings(tmp_path / e / f"cellline_{e}_embeddings.h5ad", np.random.randn(3, d), ["c1", "c2", "c3"] if e == "a" else ["c1", "c2", "c4"])
    emb = load_cell_line_embeddings(tmp_path, ["a", "b"])
    assert emb["a"]["c1"].shape == (4,) and emb["b"]["c4"].shape == (6,)
    df = pd.DataFrame({"cell_line_id": ["c1", "c2", "c3", "c4"], "drug_id": "d", "auc": 1.0, "fold": 0})
    assert keep_complete_cell_lines(df, emb)["cell_line_id"].tolist() == ["c1", "c2"]       # only cells in BOTH experts
