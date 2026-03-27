# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:29:20 2026

@author: wiewe372
"""

from pathlib import Path

from qpcr.io import load_biorad_csv
from qpcr.preprocess import filter_invalid_ct, filter_controls


def test_filter_invalid_ct_removes_missing():
    filepath = Path("tests/data/SsoAdv_Univ_SYBR_cDNA_30s_Wiebke-13-02-26.csv")
    df = load_biorad_csv(filepath, plate_id="test")

    filtered = filter_invalid_ct(df, drop_missing_ct=True)

    assert filtered["ct"].isna().sum() == 0


def test_filter_controls_removes_ntc():
    filepath = Path("tests/data/SsoAdv_Univ_SYBR_cDNA_30s_Wiebke-13-02-26.csv")
    df = load_biorad_csv(filepath, plate_id="test")

    filtered = filter_controls(df, keep_controls=False)

    assert not filtered["content"].str.upper().isin(["NTC", "NRT"]).any()