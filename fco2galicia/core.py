# -*- coding: utf-8 -*-
"""
fco2galicia.core - prediction of sea-air delta fCO2 with the fCO2GalicIA
machine-learning ensemble.

The trained model is stored in model/fCO2GalicIA/ as portable NumPy arrays
(no pickle files): prediction needs only NumPy and pandas.

Reference
  Fontela, M., & Padín, X. A. (currently under review). Air-Sea CO2 Fluxes in
  the NW Iberian Upwelling System from fCO2 observations and a
  machine-learning ensemble.

Author: Marcos Fontela (mfontela@iim.csic.es)
"""
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

OUTPUT_COLUMN = "deltafCO2"
TIME_PREDICTORS = ["year", "week_sin", "week_cos"]
WEEK_PERIOD = 365.2425 / 7.0          # mean number of ISO weeks per year

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(_PKG_DIR), "model", "fCO2GalicIA")


def add_time_predictors(df, date_col="date", week_period=WEEK_PERIOD):
    """Add year, week_sin and week_cos derived from a date column.

    week is the ISO 8601 week of the year (1-53), placed on a circle:
        week_sin = sin(2 pi (week - 0.5) / P),  week_cos = cos(2 pi (week - 0.5) / P),
    with P = 365.2425 / 7. year is the calendar year of the date.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    week = dt.dt.isocalendar().week.to_numpy(dtype=float)
    ang = 2.0 * np.pi * (week - 0.5) / float(week_period)
    out["year"] = dt.dt.year.astype(float)
    out["week_sin"] = np.sin(ang)
    out["week_cos"] = np.cos(ang)
    return out


# -----------------------------------------------------------------------------
# Evaluators of the ensemble members: Random Forest, SVR (RBF kernel), XGBoost
# -----------------------------------------------------------------------------
class _Forest:
    def __init__(self, z):
        self.right = z["right"].astype(np.int32)
        self.leaf = self.right < 0
        n = self.right.size
        self.left = np.where(self.leaf, -1, np.arange(1, n + 1)).astype(np.int32)
        self.feature = np.where(self.leaf, 0, z["feature"]).astype(np.int32)
        self.value = z["payload"].astype(np.float64)
        self.roots = z["roots"].astype(np.int32)

    def predict(self, X, chunk=50000):
        X32 = np.asarray(X, dtype=np.float32)
        out = np.empty(X32.shape[0])
        n_trees, n_feat = self.roots.size, X32.shape[1]
        for s in range(0, X32.shape[0], chunk):
            Xf = np.ascontiguousarray(X32[s:s + chunk])
            n = Xf.shape[0]
            Xf = Xf.ravel()
            node = np.tile(self.roots, n)
            rowoff = np.repeat(np.arange(n, dtype=np.int64) * n_feat, n_trees)
            act = np.flatnonzero(~self.leaf[node])
            while act.size:
                nd = node[act]
                go_left = Xf[rowoff[act] + self.feature[nd]] <= self.value[nd]
                nxt = np.where(go_left, self.left[nd], self.right[nd])
                node[act] = nxt
                act = act[~self.leaf[nxt]]
            out[s:s + chunk] = self.value[node].reshape(n, n_trees).mean(axis=1)
        return out


class _Kernel:
    def __init__(self, z):
        self.a, self.b = z["a"], z["b"]
        self.sv = z["sv"]
        self.coef = z["coef"].ravel()
        self.intercept = float(z["intercept"])
        self.gamma = float(z["gamma"])
        self.sv_sq = np.einsum("ij,ij->i", self.sv, self.sv)

    def predict(self, X, max_elements=2.0e7):
        Z = np.asarray(X, dtype=float) * self.a + self.b
        g = self.gamma
        chunk = max(1, int(max_elements // max(1, self.sv.shape[0])))
        out = np.empty(Z.shape[0])
        for s in range(0, Z.shape[0], chunk):
            z = Z[s:s + chunk]
            K = z @ self.sv.T
            K *= 2.0 * g
            K -= (g * np.einsum("ij,ij->i", z, z))[:, None]
            K -= (g * self.sv_sq)[None, :]
            np.minimum(K, 0.0, out=K)
            np.exp(K, out=K)
            out[s:s + chunk] = K @ self.coef + self.intercept
        return out


class _Boosted:
    def __init__(self, z):
        self.left = z["left"].astype(np.int64)
        self.right = z["right"].astype(np.int64)
        self.leaf = self.left < 0
        self.feature = np.where(self.leaf, 0, z["feature"]).astype(np.int64)
        self.threshold = z["split_condition"].astype(np.float32)
        self.roots = z["roots"].astype(np.int64)
        self.base = np.float32(z["base"])
        self.a, self.b = z["a"], z["b"]

    def predict(self, X, chunk=50000):
        Xr = (np.asarray(X, dtype=float) * self.a + self.b).astype(np.float32)
        out = np.empty(Xr.shape[0])
        for s in range(0, Xr.shape[0], chunk):
            Xc = Xr[s:s + chunk]
            n = Xc.shape[0]
            node = np.broadcast_to(self.roots, (n, self.roots.size)).copy()
            rows = np.arange(n)[:, None]
            while True:
                active = ~self.leaf[node]
                if not active.any():
                    break
                go_left = Xc[rows, self.feature[node]] < self.threshold[node]
                node = np.where(active, np.where(go_left, self.left[node], self.right[node]), node)
            leaves = self.threshold[node]
            acc = np.full(n, self.base, dtype=np.float32)
            for t in range(leaves.shape[1]):
                acc += leaves[:, t]
            out[s:s + chunk] = acc
        return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Ensemble
# -----------------------------------------------------------------------------
class FCO2GalicIA:
    """fCO2GalicIA ensemble: mean of Random Forest, SVR and XGBoost.

    Parameters
    ----------
    model_dir : str, optional
        Folder with manifest.json and the model files (default model/fCO2GalicIA).
    verify : bool
        Check the SHA-256 checksum of every model file.
    """

    def __init__(self, model_dir=None, verify=True):
        model_dir = model_dir or os.environ.get("FCO2GALICIA_MODEL_DIR") or DEFAULT_MODEL_DIR
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        files = self.manifest["files"]
        if verify:
            for meta in files.values():
                p = os.path.join(model_dir, meta["file"])
                if _sha256(p) != meta["sha256"]:
                    raise IOError("Checksum mismatch for %s: file corrupted or modified "
                                  "(or a Git LFS pointer; run 'git lfs pull')." % p)
        self.predictors = list(self.manifest["predictors"])
        self.week_period = float(self.manifest["time_encoding"]["week_period"])
        self._members = []
        for key, cls in (("random_forest", _Forest), ("svr", _Kernel), ("xgboost", _Boosted)):
            with np.load(os.path.join(model_dir, files[key]["file"])) as z:
                self._members.append(cls(z))

    @property
    def version(self):
        return self.manifest.get("model_version")

    def predict(self, df, date_col="date", warn=True):
        """Return a copy of `df` with the column deltafCO2 (uatm) appended.

        `df` holds the environmental predictors, lat, lon and a date column
        (year, week_sin and week_cos are derived from it). Rows with a missing
        predictor get NaN.
        """
        if any(c not in df.columns for c in TIME_PREDICTORS) and date_col in df.columns:
            df = add_time_predictors(df, date_col, self.week_period)
        missing = [c for c in self.predictors if c not in df.columns]
        if missing:
            raise KeyError("Missing required columns: %s" % ", ".join(missing))
        X = df[self.predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        for p, cap in self.manifest.get("input_caps", {}).items():   # as in training
            j = self.predictors.index(p)
            X[:, j] = np.where(X[:, j] > cap, cap, X[:, j])
        out = df.copy()
        out[OUTPUT_COLUMN] = self.predict_matrix(X, warn=warn)
        return out

    def predict_matrix(self, X, warn=True):
        """deltafCO2 (uatm) from an (n, p) array whose columns follow self.predictors."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.predictors):
            raise ValueError("X must have %d columns: %s" % (len(self.predictors), self.predictors))
        ok = np.all(np.isfinite(X), axis=1)
        y = np.full(X.shape[0], np.nan)
        if ok.any():
            if warn:
                self._check_inputs(X[ok])
            y[ok] = np.mean([m.predict(X[ok]) for m in self._members], axis=0)
        return y

    def predict_columns(self, columns, dates):
        """Bridge for R (reticulate): `columns` is a named list of numeric
        vectors, `dates` a vector of 'YYYY-MM-DD' strings. Returns a list."""
        df = pd.DataFrame({k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in columns.items()})
        df["date"] = list(np.atleast_1d(dates))
        return self.predict(df)[OUTPUT_COLUMN].astype(float).tolist()

    def self_test(self, atol=1e-6):
        """Reproduce the reference predictions stored in the manifest."""
        ref = self.manifest["reference_predictions"]
        d = float(np.max(np.abs(self.predict_matrix(np.asarray(ref["X"], float), warn=False)
                                - np.asarray(ref[OUTPUT_COLUMN], float))))
        if d > atol:
            raise AssertionError("self-test failed: max |difference| = %g uatm" % d)
        return d

    def _check_inputs(self, X):
        rng = self.manifest["valid_input_ranges"]
        n_out, n_rows = 0, np.zeros(X.shape[0], dtype=bool)
        for j, p in enumerate(self.predictors):
            if p in rng:
                bad = (X[:, j] < rng[p][0]) | (X[:, j] > rng[p][1])
                n_rows |= bad
        if n_rows.any():
            warnings.warn("%d of %d rows have predictors outside the valid input ranges "
                          "(see manifest.json): extrapolation." % (int(n_rows.sum()), X.shape[0]),
                          stacklevel=3)

    def __repr__(self):
        return "<fCO2GalicIA %s | deltafCO2 ~ %s>" % (self.version, " + ".join(self.predictors))


def load_model(model_dir=None, verify=True):
    """Load the fCO2GalicIA ensemble."""
    return FCO2GalicIA(model_dir=model_dir, verify=verify)
