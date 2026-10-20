# -*- coding: utf-8 -*-
"""Tests of the fco2galicia package (from the repository folder: pytest -q).
Author: Marcos Fontela"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fco2galicia as fg  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return fg.load_model()


def example():
    return pd.read_csv(os.path.join(ROOT, "examples", "input_example.csv"))


def test_self_test(model):
    assert model.self_test() < 1e-6


def test_time_predictors():
    d = fg.add_time_predictors(pd.DataFrame({"date": ["2010-01-03", "2010-01-04", "2021-01-01"]}))
    ang = 2 * np.pi * (np.array([53, 1, 53]) - 0.5) / (365.2425 / 7)     # ISO 8601 weeks
    assert np.allclose(d["week_sin"], np.sin(ang)) and np.allclose(d["week_cos"], np.cos(ang))
    assert d["year"].tolist() == [2010, 2010, 2021]


def test_row_order_invariance(model):
    df = example()
    a = model.predict(df)["deltafCO2"].to_numpy()
    b = model.predict(df.iloc[::-1])["deltafCO2"].to_numpy()[::-1]
    assert np.allclose(a, b, atol=1e-10)


def test_missing_values_propagate(model):
    df = example()
    df.loc[3, "CHL"] = np.nan
    out = model.predict(df)
    assert np.isnan(out.loc[3, "deltafCO2"]) and out.drop(index=3)["deltafCO2"].notna().all()


def test_missing_column(model):
    with pytest.raises(KeyError):
        model.predict(example().drop(columns="UI_7"))


def test_chlorophyll_cap(model):
    df = example().head(1)
    a = model.predict(df.assign(CHL=15.0), warn=False)["deltafCO2"].iloc[0]
    b = model.predict(df.assign(CHL=40.0), warn=False)["deltafCO2"].iloc[0]
    assert a == b


def test_r_bridge(model):
    df = example()
    vars_ = [p for p in model.predictors if p not in fg.TIME_PREDICTORS]
    res = model.predict_columns({p: df[p].tolist() for p in vars_}, df["date"].tolist())
    assert np.allclose(res, model.predict(df)["deltafCO2"], atol=1e-12)


def test_range_warning(model):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.predict(example().head(1).assign(lat=45.0))
    assert any("valid input ranges" in str(x.message) for x in w)
