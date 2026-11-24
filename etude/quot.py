"""Verification sur 10 ans (barres journalieres ajustees) des scenarios d'ecart a l'ouverture : achat a l'ouverture, resultat a la cloture.
Apprentissage 2016-2022, controle 2023-2026. Sortie : quot.parquet et quot_scen.csv"""
import glob, os, numpy as np, pandas as pd, datetime
from concurrent.futures import ProcessPoolExecutor
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
def un(p):
    try: A = np.load(p)["a"]
    except Exception: return None
    if len(A) < 30: return None
    t, o, h, l, c, v = A.T
    pc = np.r_[np.nan, c[:-1]]
    d = pd.DataFrame({"sym": os.path.basename(p)[:-4], "t": t.astype(np.int64), "o": o, "h": h, "l": l, "c": c, "pc": pc,
                      "dv20": pd.Series(c * v).rolling(20).mean().shift(1).values, "r1j": (pc / np.r_[np.nan, np.nan, c[:-2]] - 1) * 100,
                      "nc": np.r_[c[1:], np.nan]})
    return d
if __name__ == "__main__":
    F = sorted(glob.glob(f"{D}/data1d/*.npz"))
    with ProcessPoolExecutor(2) as ex: L = [x for x in ex.map(un, F, chunksize=20) if x is not None]
    df = pd.concat(L, ignore_index=True)
    df["d"] = pd.to_datetime(df.t, unit="s").dt.strftime("%Y-%m-%d"); df["an"] = df.d.str[:4].astype(int)
    df = df[(df.pc >= 1) & (df.dv20 >= 1e6) & (df.o > 0)].copy()
    df["gap"] = (df.o / df.pc - 1) * 100; df = df[df.gap.between(-60, 300)]
    df["rc"] = (df.c / df.o - 1) * 100; df["hit3"] = df.h >= df.o * 1.03; df["low3"] = df.l <= df.o * 0.97
    df["rn"] = (df.nc / df.c - 1) * 100  # lendemain (cloture a cloture)
    df.to_parquet(f"{D}/quot.parquet"); print(len(F), "actions", len(df), "seances", df.an.min(), "-", df.an.max())
