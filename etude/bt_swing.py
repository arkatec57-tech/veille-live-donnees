"""Rejeu des règles du robot swing (2 à 5 jours) sur 10 ans de cours quotidiens ajustés.
Signal calculé à la clôture du jour J, achat à l'ouverture de J+1 (prix + 0,05 % d'écart), sortie selon la règle ; frais exclus.
Deux périodes : P1 = 2016-11 à 2021-06, P2 = 2021-07 à 2026-11. Une règle n'est retenue que si elle gagne sur les deux.
Sortie : trades.parquet (une ligne par trade simulé, toutes règles) puis synthèse par règle.
"""
import os, sys, json, glob, numpy as np, pandas as pd
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
SLIP = 0.0005
COUPURE = pd.Timestamp("2021-07-01").timestamp()

def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0)); out = np.full(len(x), np.nan); out[n - 1:] = (c[n:] - c[:-n]) / n; return out
def rsi(c, n):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values; ad = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))
def atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(n).mean().values

# règles : signal (masque booléen sur J) + sortie
def signaux(o, h, l, c, v):
    s200, s50, s20, s5 = sma(c, 200), sma(c, 50), sma(c, 20), sma(c, 5)
    r2, r14, a14 = rsi(c, 2), rsi(c, 14), atr(h, l, c)
    dv = sma(c * v, 20); hi20 = pd.Series(h).rolling(20).max().shift(1).values; vm = sma(v, 20)
    ret3 = c / np.roll(c, 3) - 1; ret3[:3] = np.nan
    base = (c >= 5) & (dv >= 20e6)
    tend = (c > s200) & (s50 > s200)
    S = {
        "repli_rsi2": base & tend & (r2 < 10),
        "repli_rsi2_fort": base & tend & (r2 < 5) & (c > s50 * 0.97),
        "repli_3j": base & tend & (ret3 <= -0.06),
        "cassure_20j": base & tend & (c > hi20) & (v > 1.8 * vm) & (r14 < 75),
        "rebond_survente": base & (c > s200) & (r14 < 30),
    }
    return S, dict(s5=s5, a14=a14)

SORTIES = {  # (stop en ATR, objectif en ATR ou None, sortie sur moyenne 5 jours, durée max en séances)
    "repli_rsi2": (2.0, None, True, 5), "repli_rsi2_fort": (2.0, None, True, 5), "repli_3j": (2.0, 3.0, True, 5),
    "cassure_20j": (1.5, 3.0, False, 5), "rebond_survente": (2.0, 3.0, True, 5),
}

def rejouer(sym, A):
    t, o, h, l, c, v = (A[:, i] for i in range(6))
    if len(c) < 260: return []
    S, X = signaux(o, h, l, c, v); out = []
    for k, m in S.items():
        stopA, objA, sur5, nmax = SORTIES[k]; idx = np.where(m[:-1])[0]; libre = 0
        for i in idx:
            if i < libre or i + 1 >= len(c) or not np.isfinite(X["a14"][i]): continue
            e = o[i + 1] * (1 + SLIP); a = X["a14"][i]; stp = e - stopA * a; obj = e + objA * a if objA else None; ps = None; j = i + 1
            while j < len(c) and j <= i + nmax:
                if l[j] <= stp: ps = min(o[j], stp) if o[j] < stp else stp; break
                if obj and h[j] >= obj: ps = max(o[j], obj); break
                if sur5 and c[j] > X["s5"][j] and j > i + 1: ps = c[j]; break
                j += 1
            if ps is None: j = min(i + nmax, len(c) - 1); ps = c[j]
            ps *= (1 - SLIP); out.append((sym, k, t[i + 1], j - i, (ps / e - 1) * 100, a / c[i] * 100)); libre = j + 1
    return out

def main():
    F = sorted(glob.glob(f"{D}/data1d/*.npz")); T = []
    for n, f in enumerate(F):
        try: T += rejouer(os.path.basename(f)[:-4], np.load(f)["a"])
        except Exception as e: pass
        if n % 500 == 0: print(n, len(F), len(T), flush=True)
    df = pd.DataFrame(T, columns=["sym", "regle", "t", "jours", "ret", "atrp"]); df.to_parquet(f"{D}/bt_swing.parquet")
    df["per"] = np.where(df.t < COUPURE, "P1", "P2"); res = {}
    for (k, p), g in df.groupby(["regle", "per"]):
        G, P0 = g[g.ret > 0], g[g.ret <= 0]
        res.setdefault(k, {})[p] = dict(n=len(g), win=round(len(G) / len(g) * 100, 1), esp=round(g.ret.mean(), 3), med=round(g.ret.median(), 3),
            mg=round(G.ret.mean(), 2) if len(G) else None, mp=round(P0.ret.mean(), 2) if len(P0) else None, pf=round(G.ret.sum() / max(1e-9, -P0.ret.sum()), 2), jours=round(g.jours.mean(), 1))
    json.dump(res, open(f"{D}/bt_swing.json", "w"), indent=1); print(json.dumps(res, indent=1))
if __name__ == "__main__": main()
