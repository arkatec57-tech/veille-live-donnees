"""Robot day trading : écarts d'ouverture à la baisse qui se referment (mesuré dans l'étude des scénarios).
Achat au premier cours de la séance, objectif / stop, sinon vente à la clôture. Variantes choisies sur P1, confirmées sur P2."""
import os, glob, json, numpy as np, pandas as pd
from bt_day import jours, COUP
D = os.path.dirname(os.path.abspath(__file__))
Q = {j[0]: (j[2][0], j[5][-1]) for j in jours(np.load(f"{D}/data1h/QQQ.npz")["a"])}
Qd = sorted(Q); Qgap = {}
for a, b in zip(Qd[:-1], Qd[1:]): Qgap[b] = (Q[b][0] / Q[a][1] - 1) * 100
VAR = {}
for g in ((-10, -5), (-5, -2), (-10, -2)):
    for pmin in (5, 20):
        for obj in (None, 3.0, 5.0):
            for stp in (None, 4.0):
                VAR[f"g{g[0]}_{g[1]}_p{pmin}_o{obj}_s{stp}"] = (g, pmin, obj, stp, None)
VAR["g-10_-5_p5_o3.0_sNone_qqq+"] = ((-10, -5), 5, 3.0, None, "+"); VAR["g-10_-5_p5_o3.0_sNone_qqq-"] = ((-10, -5), 5, 3.0, None, "-")
T = []
for f in sorted(glob.glob(f"{D}/data1h/*.npz")):
    sym = os.path.basename(f)[:-4]
    try: J = jours(np.load(f)["a"])
    except Exception: continue
    if len(J) < 30: continue
    closes = [j[5][-1] for j in J]; dvs = [float((j[5] * j[6]).sum()) for j in J]
    for i in range(21, len(J)):
        dd, t0, o, h, l, c, v = J[i]; pc = closes[i - 1]; dv20 = np.mean(dvs[i - 20:i])
        if dv20 < 20e6 or pc < 5: continue
        gap = (o[0] / pc - 1) * 100
        if gap > -2 or gap < -10: continue
        qg = Qgap.get(dd)
        for k, (g, pmin, obj, stp, qf) in VAR.items():
            if not (g[0] < gap <= g[1]) or o[0] < pmin: continue
            if qf == "+" and not (qg is not None and qg > 0.3): continue
            if qf == "-" and not (qg is not None and qg < -0.3): continue
            for sl in (0.0005, 0.0015):
                e = o[0] * (1 + sl); ps = None
                for b in range(len(c)):
                    if stp and l[b] <= e * (1 - stp / 100): ps = min(o[b], e * (1 - stp / 100)) if b else e * (1 - stp / 100); break
                    if obj and h[b] >= e * (1 + obj / 100): ps = e * (1 + obj / 100); break
                if ps is None: ps = c[-1]
                T.append((sym, k, sl, t0, gap, (ps * (1 - sl) / e - 1) * 100))
df = pd.DataFrame(T, columns=["sym", "k", "sl", "t", "gap", "ret"]); df["per"] = np.where(df.t < COUP, "P1", "P2"); df.to_parquet(f"{D}/bt_day2.parquet")
R0 = {}
for (k, sl, p), g in df.groupby(["k", "sl", "per"]):
    G = g[g.ret > 0]; R0.setdefault(f"{k}|{sl}", {})[p] = dict(n=len(g), win=round(len(G) / len(g) * 100, 1), esp=round(g.ret.mean(), 3), pf=round(G.ret.sum() / max(1e-9, -g[g.ret <= 0].ret.sum()), 2))
json.dump(R0, open(f"{D}/bt_day2.json", "w"), indent=1)
for k, v in sorted(R0.items(), key=lambda kv: -kv[1].get("P1", {}).get("esp", -9))[:24]: print(f"{k:42s}", v.get("P1"), "|", v.get("P2"))
