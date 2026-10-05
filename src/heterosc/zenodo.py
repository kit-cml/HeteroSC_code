"""Download and unpack the HeteroSC data record from Zenodo (https://doi.org/10.5281/zenodo.23153537).

The record holds three archives (annotation, drug_response, gene_perturbation). Raw GDSC dose-response files are
*not* on Zenodo; see :func:`heterosc.gdsc.download_gdsc`.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

ZENODO_DOI = "10.5281/zenodo.23153537"
RECORD_ID = "23153537"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"
FILES: Dict[str, str] = {                                   # archive -> MD5 (as shown on the Zenodo record)
    "annotation.rar": "0cb08bcad13e54b0b42492b929bbd097",
    "drug_response.rar": "43fec0e20632007264c238c7a04029f5",
    "gene_perturbation.rar": "7b50c606d74eb7dddc499a882eb5ee74",
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, md5: Optional[str] = None, overwrite: bool = False, verbose: bool = True) -> Path:
    """Download ``url`` to ``dest`` (resumable only by re-download) and verify its MD5 if given."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and (md5 is None or md5sum(dest) == md5):
        if verbose:
            print(f"[skip] {dest.name} already present and verified")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if verbose:
        print(f"[download] {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if md5 is not None and md5sum(tmp) != md5:
        tmp.unlink()
        raise IOError(f"MD5 mismatch for {dest.name}: the download is corrupt or the record changed")
    tmp.replace(dest)
    return dest


def _extract_rar(path: Path, out: Path) -> None:
    for exe, args in (("7z", ["x", "-y", f"-o{out}"]), ("7zz", ["x", "-y", f"-o{out}"]), ("7za", ["x", "-y", f"-o{out}"]),
                      ("unrar", ["x", "-y", "-idq"]), ("bsdtar", ["-xf"])):
        if shutil.which(exe):
            cmd = [exe] + args + ([str(path), str(out)] if exe == "unrar" else [str(path)] if exe != "bsdtar" else [str(path), "-C", str(out)])
            subprocess.run(cmd, check=True)
            return
    try:
        import rarfile
        with rarfile.RarFile(path) as rf:
            rf.extractall(out)
        return
    except ImportError:
        pass
    raise RuntimeError("Cannot open .rar: install 7-Zip (7z) or unrar, or `pip install rarfile` plus unrar.")


def extract_archive(archive: Path, dest_root: Path, name: Optional[str] = None) -> Path:
    """Extract ``archive`` into ``dest_root/<name>/`` whether or not it already contains a top-level folder."""
    archive, dest_root = Path(archive), Path(dest_root)
    name = name or archive.stem
    target = dest_root / name
    with tempfile.TemporaryDirectory(dir=dest_root if dest_root.exists() else None) as tmp:
        tmp = Path(tmp)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        elif archive.suffix == ".rar":
            _extract_rar(archive, tmp)
        else:
            raise ValueError(f"unsupported archive type: {archive.suffix}")
        items = list(tmp.iterdir())
        src = items[0] if len(items) == 1 and items[0].is_dir() and items[0].name == name else tmp
        if target.exists():
            shutil.rmtree(target)
        dest_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
    return target


def download_data(dest: Path = Path("data"), parts: Optional[Iterable[str]] = None, extract: bool = True,
                  keep_archives: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """Download (and extract) parts of the Zenodo record into ``dest``.

    ``parts``: subset of ``{"annotation", "drug_response", "gene_perturbation"}`` (default: all).
    Returns ``{part: extracted_folder}``.
    """
    dest = Path(dest)
    parts = list(parts) if parts else [f[:-4] for f in FILES]
    out = {}
    for part in parts:
        fname = f"{part}.rar"
        if fname not in FILES:
            raise KeyError(f"unknown part {part!r}; choose from {[f[:-4] for f in FILES]}")
        arch = download_file(f"{BASE_URL}/{fname}?download=1", dest / "_archives" / fname, FILES[fname], verbose=verbose)
        if extract:
            out[part] = extract_archive(arch, dest, part)
            if verbose:
                print(f"[ok] {part} -> {out[part]}")
            if not keep_archives:
                arch.unlink()
    return out
