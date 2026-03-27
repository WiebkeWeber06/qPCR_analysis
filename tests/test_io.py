# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:54:18 2026

@author: wiewe372
"""

from pathlib import Path

from qpcr.io import load_biorad_csv


def test_load_biorad_csv():
    filepath = Path("tests/data/SsoAdv_Univ_SYBR_cDNA_30s_Wiebke-13-02-26.csv")
    df = load_biorad_csv(filepath, plate_id="test_plate")

    assert len(df) > 0
    assert "plate_id" in df.columns
    assert "well" in df.columns
    assert "target" in df.columns
    assert "sample_id" in df.columns
    assert "ct" in df.columns