# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:03:02 2026

@author: wiewe372
"""

from pathlib import Path

from qpcr.io import load_biorad_csv
from qpcr.schema import validate_qpcr_dataframe


def test_validate_qpcr_dataframe():
    filepath = Path("tests/data/SsoAdv_Univ_SYBR_cDNA_30s_Wiebke-13-02-26.csv")
    df = load_biorad_csv(filepath, plate_id="test_plate")

    validated = validate_qpcr_dataframe(df)

    assert validated is not None
    assert "ct" in validated.columns
    assert "content" in validated.columns