import numpy as np
import pandas as pd

GAS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
FLOOR_PPM = 1.0   


def build_features(gases: pd.DataFrame) -> pd.DataFrame:
    """Raw gas ppm (5 columns) -> model features (17 columns)."""
    g = gases[GAS].clip(lower=FLOOR_PPM)
    feat = pd.DataFrame(index=gases.index)

    for c in GAS:
        feat[f"log_{c}"] = np.log10(g[c])

    total = g.sum(axis=1)
    feat["log_total"] = np.log10(total)

    for c in GAS:
        feat[f"pct_{c}"] = g[c] / total

    tri = g[["CH4", "C2H4", "C2H2"]]
    tri_pct = tri.div(tri.sum(axis=1), axis=0)
    for c in tri_pct.columns:
        feat[f"tri_{c}"] = tri_pct[c]

    feat["r_C2H2_C2H4"] = np.log10(g["C2H2"] / g["C2H4"])
    feat["r_CH4_H2"] = np.log10(g["CH4"] / g["H2"])
    feat["r_C2H4_C2H6"] = np.log10(g["C2H4"] / g["C2H6"])
    return feat