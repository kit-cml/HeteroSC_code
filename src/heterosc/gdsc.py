"""Download raw GDSC dose-response tables from the Sanger FTP (not redistributed on Zenodo).

GDSC data are used under the GDSC data usage policy (https://depmap.sanger.ac.uk/documentation/data-usage-policy/).
Record the release/date you download. The processed response table in the Zenodo record is enough to run the tutorials;
you only need these files to rebuild the table from scratch.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

URL = "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/"
TARGETS = {
    "GDSC1_fitted_dose_response.csv": r"GDSC1_fitted_dose_response.*\.csv",
    "GDSC2_fitted_dose_response.csv": r"GDSC2_fitted_dose_response.*\.csv",
    "screened_compounds.csv": r"screened_compounds.*\.csv",
}


def download_gdsc(out_dir: Path, verbose: bool = True) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "ignore")
    names = re.findall(r'href="([^"]+\.(?:csv|xlsx))"', html)
    done = []
    for dest, pattern in TARGETS.items():
        hits = [n for n in names if re.match(pattern, n)]
        if not hits:
            print(f"[warn] no file matching {pattern} in the current GDSC release")
            continue
        path = out_dir / dest
        if not path.exists():
            with urllib.request.urlopen(URL + hits[0], timeout=300) as r, open(path, "wb") as f:
                f.write(r.read())
        if verbose:
            print(f"[ok] {dest} <- {hits[0]}")
        done.append(path)
    print("Write down the GDSC release and download date in your notes.")
    return done
