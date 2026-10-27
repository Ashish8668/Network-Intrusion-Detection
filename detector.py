import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Detector:
    def __init__(self, path="rbm_artifacts.pkl"):
        artifact_path = Path(path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with artifact_path.open("rb") as f:
                a = pickle.load(f)
        self.W, self.b, self.c = a["W"], a["b"], a["c"]
        self.threshold = float(a["threshold"])
        self.scaler = a["scaler"]
        self.feature_names = a["feature_names"]
        self.group_names = a["group_names"]
        self.groups = a["groups"]
        self.mu, self.sd = a["mu"], a["sd"]
        self.cat_levels = a["cat_levels"]
        self.num_cols = a["num_cols"]
        self.log_cols = a["log_cols"]
        self.cols = a["cols"]
        self.cause_rules = a["cause_rules"]

    def to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.num_cols if col not in df.columns]
        missing += [col for col in self.cat_levels if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        X = df[self.num_cols].astype(float).copy()
        for col in self.log_cols:
            X[col] = np.log1p(X[col])

        onehots = {}
        for col, levels in self.cat_levels.items():
            for lv in levels:
                onehots[f"{col}={lv}"] = (df[col] == lv).astype(float)

        X = pd.concat([X, pd.DataFrame(onehots, index=df.index)], axis=1)
        X = X[self.feature_names]
        return np.clip(self.scaler.transform(X), 0, 1).astype(np.float32)

    def free_energy(self, v):
        return -(v @ self.b) - np.logaddexp(0, v @ self.W + self.c).sum(axis=1)

    def reconstruct(self, v):
        h = _sigmoid(v @ self.W + self.c)
        return _sigmoid(h @ self.W.T + self.b)

    def feat_err(self, v):
        e = (v - self.reconstruct(v)) ** 2
        return np.stack([e[:, ix].sum(1) for ix in self.groups.values()], axis=1)

    def explain(self, v_row, top_k=4):
        z = (self.feat_err(v_row[None, :])[0] - self.mu) / self.sd
        order = np.argsort(-z)[:top_k]
        top = [(self.group_names[i], round(float(z[i]), 2)) for i in order]
        names = {t[0] for t in top}
        best, score = "Unknown / unusual traffic", 0
        for feats, cause in self.cause_rules:
            s = len(names & feats)
            if s > score:
                best, score = cause, s
        return top, best

    def predict_df(self, df: pd.DataFrame):
        V = self.to_matrix(df)
        fe = self.free_energy(V)
        out = []
        for i in range(len(V)):
            row = {
                "anomaly": bool(fe[i] > self.threshold),
                "free_energy": round(float(fe[i]), 3),
                "threshold": round(float(self.threshold), 3),
            }
            if row["anomaly"]:
                top, cause = self.explain(V[i])
                row["top_features"] = [{"feature": f, "z_score": z} for f, z in top]
                row["probable_cause"] = cause
            else:
                row["top_features"] = []
                row["probable_cause"] = ""
            out.append(row)
        return out
