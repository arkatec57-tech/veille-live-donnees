"""Exploration autour du « rebond en survente dans une tendance de fond » : choix sur P1, confirmation sur P2."""
import os, glob, json, numpy as np, pandas as pd
from bt_swing import sma, rsi, atr, SLIP, COUPURE
D = os.path.dirname(os.path.abspath(__file__))
spy = np.load(f"{D}/data1d/SPY.npz")["a"]; spy_t = spy[:, 0]; spy_c = spy[:, 4]; spy_ok = spy_c > sma(spy_c, 200)
spy_map = dict(zip((spy_t // 86400).astype(int), spy_ok))
VAR = {}
for rs in (25, 30, 35):
    for stopA in (1.5, 2.0, 3.0):
        for nmax in (3, 5, 7):
            for sortie in ("s5", "obj"):
                VAR[f"r{rs}_s{stopA}_n{nmax}_{sortie}"] = (rs, stopA, nmax, sortie)
VAR["base_aleat"] = (None, 2.0, 5, "s5")
def rejouer(sym, A, out):
    t, o, h, l, c, v = (A[:, i] for i in range(6))
    if len(c) < 260: return
    s200, s5 = sma(c, 200), sma(c, 5); r14 = rsi(c, 14); a14 = atr(h, l, c); dv = sma(c * v, 20)
    base = (c >= 5) & (dv >= 20e6) & (c > s200)
    mk = np.array([spy_map.get(int(x // 86400), False) for x in t])
    for k, (rs, stopA, nmax, sortie) in VAR.items():
        m = base & (r14 < rs) if rs else base & (np.arange(len(c)) % 7 == 0)
        idx = np.where(m[:-1])[0]; libre = 0
        for i in idx:
            if i < libre or not np.isfinite(a14[i]): continue
            e = o[i + 1] * (1 + SLIP); a = a14[i]; stp = e - stopA * a; obj = e + 3 * a if sortie == "obj" else None; ps = None; j = i + 1
            while j < len(c) and j <= i + nmax:
                if l[j] <= stp: ps = min(o[j], stp); break
                if obj and h[j] >= obj: ps = max(o[j], obj); break
                if sortie == "s5" and j > i + 1 and c[j] > s5[j]: ps = c[j]; break
                j += 1
            if ps is None: j = min(i + nmax, len(c) - 1); ps = c[j]
            out.append((sym, k, t[i + 1], j - i, (ps * (1 - SLIP) / e - 1) * 100, bool(mk[i]), r14[i])); libre = j + 1
T = []
for n, f in enumerate(sorted(glob.glob(f"{D}/data1d/*.npz"))):
    try: rejouer(os.path.basename(f)[:-4], np.load(f)["a"], T)
    except Exception: pass
df = pd.DataFrame(T, columns=["sym", "k", "t", "j", "ret", "mk", "r14"]); df["per"] = np.where(df.t < COUPURE, "P1", "P2"); df.to_parquet(f"{D}/bt_swing2.parquet")
def st(g): G = g[g.ret > 0]; P0 = g[g.ret <= 0]; return dict(n=len(g), win=round(len(G) / max(1, len(g)) * 100, 1), esp=round(g.ret.mean(), 3), pf=round(G.ret.sum() / max(1e-9, -P0.ret.sum()), 2), j=round(g.j.mean(), 1))
R0 = {}
for k, g in df.groupby("k"):
    R0[k] = {p: st(g[g.per == p]) for p in ("P1", "P2")}
    R0[k]["P1_marche_ok"] = st(g[(g.per == "P1") & g.mk]); R0[k]["P2_marche_ok"] = st(g[(g.per == "P2") & g.mk])
json.dump(R0, open(f"{D}/bt_swing2.json", "w"), indent=1)
L = sorted(R0.items(), key=lambda kv: -kv[1]["P1"]["esp"])
for k, v in L[:20] + [("base_aleat", R0["base_aleat"])]:
    print(f"{k:22s} P1 {v['P1']} | P2 {v['P2']} | P1m {v['P1_marche_ok']['esp']} P2m {v['P2_marche_ok']['esp']} n2m {v['P2_marche_ok']['n']}")
