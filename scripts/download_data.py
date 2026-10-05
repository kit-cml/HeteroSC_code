#!/usr/bin/env python
"""Download the HeteroSC data from Zenodo (https://doi.org/10.5281/zenodo.23153537), verify MD5, and extract.

  python scripts/download_data.py --dest data                     # everything (~650 MB download)
  python scripts/download_data.py --dest data --parts annotation  # only the annotation datasets
  python scripts/download_data.py --gdsc data/gdsc                # raw GDSC tables from Sanger (optional)
"""
import argparse
from pathlib import Path

from heterosc.gdsc import download_gdsc
from heterosc.zenodo import download_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--parts", nargs="+", choices=["annotation", "drug_response", "gene_perturbation"])
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--gdsc", type=Path, default=None, help="also download raw GDSC tables into this folder")
    a = ap.parse_args()
    download_data(a.dest, a.parts, keep_archives=a.keep_archives)
    if a.gdsc:
        download_gdsc(a.gdsc)


if __name__ == "__main__":
    main()
