"""Robot swing retenu : rebond en survente (RSI 14 < 30) au-dessus de la moyenne 200 jours, actions liquides (>= 5 $, >= 20 M$/jour).
Achat à l'ouverture suivante, stop 3 ATR, sortie à la première clôture au-dessus de la moyenne 5 jours, au plus tard 7 séances.
Simulation de portefeuille : 10 000 $ fictifs, 5 positions au plus (2 000 $ chacune), les signaux les plus survendus d'abord. Frais exclus."""
import os, glob, json, numpy as np, pandas as pd
from bt_swing import sma, rsi, atr, SLIP, COUPURE
D = os.path.dirname(os.path.abspath(__file__)); RS, STOPA, NMAX = 30, 3.0, 7
T = []
for f in sorted(glob.glob(f"{D}/data1d/*.npz")):
    try:
        A = np.load(f)["a"]; t, o, h, l, c, v = (A[:, i] for i in range(6))
        if len(c) < 260: continue
        s200, s5 = sma(c, 200), sma(c, 5); r14 = rsi(c, 14); a14 = atr(h, l, c); dv = sma(c * v, 20)
        m = (c >= 5) & (dv >= 20e6) & (c > s200) & (r14 < RS); libre = 0
        for i in np.where(m[:-1])[0]:
            if i < libre or not np.isfinite(a14[i]): continue
            e = o[i + 1] * (1 + SLIP); stp = e - STOPA * a14[i]; ps = None; j = i + 1
            while j < len(c) and j <= i + NMAX:
                if l[j] <= stp: ps = min(o[j], stp); why = "stop"; break
                if j > i + 1 and c[j] > s5[j]: ps = c[j]; why = "moy5"; break
                j += 1
            if ps is None: j = min(i + NMAX, len(c) - 1); ps = c[j]; why = "duree"
            T.append((os.path.basename(f)[:-4], t[i + 1], t[j], (ps * (1 - SLIP) / e - 1) * 100, r14[i], why, dv[i], c[i] / s200[i], a14[i] / c[i])); libre = j + 1
    except Exception: pass
df = pd.DataFrame(T, columns=["sym", "t0", "t1", "ret", "rsi", "why", "dv", "ext", "atrp"])
import sys
TRI = os.environ.get("TRI", "rsi"); MAXP = int(os.environ.get("MAXP", "5"))
cle = {"rsi": "rsi", "dv": "-dv", "ext": "-ext", "atrp": "atrp", "hasard": "h"}[TRI]
df["h"] = np.random.default_rng(1).random(len(df)); df["-dv"] = -df.dv; df["-ext"] = -df.ext
df = df.sort_values(["t0", cle])
cash, pos, eq = 10000.0, [], []
jours = sorted(set(df.t0) | set(df.t1)); par = {k: g for k, g in df.groupby("t0")}; pris = []
for d in jours:
    for p in [p for p in pos if p["t1"] <= d]: cash += p["m"] * (1 + p["ret"] / 100); pos.remove(p)
    for _, x in (par.get(d, pd.DataFrame()).iterrows()):
        if len(pos) >= MAXP: break
        val = cash + sum(p["m"] for p in pos); m = min(cash, val / MAXP)
        if m < 100: break
        cash -= m; pos.append(dict(t1=x.t1, m=m, ret=x.ret)); pris.append(x)
    eq.append((d, cash + sum(p["m"] for p in pos)))
E = pd.DataFrame(eq, columns=["t", "v"]); E["d"] = pd.to_datetime(E.t, unit="s")
spy = np.load(f"{D}/data1d/SPY.npz")["a"]; S = pd.Series(spy[:, 4], index=pd.to_datetime(spy[:, 0], unit="s"))
def resume(E0):
    v = E0.v.values; ans = (E0.d.iloc[-1] - E0.d.iloc[0]).days / 365.25; dd = (v / np.maximum.accumulate(v) - 1).min() * 100
    s0 = S[S.index >= E0.d.iloc[0]].iloc[0]; s1 = S[S.index <= E0.d.iloc[-1]].iloc[-1]
    return dict(annuel=round(((v[-1] / v[0]) ** (1 / ans) - 1) * 100, 1), pire_baisse=round(dd, 1), sp500_annuel=round(((s1 / s0) ** (1 / ans) - 1) * 100, 1))
P = pd.DataFrame(pris); P["per"] = np.where(P.t0 < COUPURE, "P1", "P2")
out = {"regle": dict(rsi=RS, stop_atr=STOPA, max_seances=NMAX, liquidite_musd=20, prix_min=5), "tous_signaux": {}, "portefeuille": {}}
for p, g in df.assign(per=np.where(df.t0 < COUPURE, "P1", "P2")).groupby("per"):
    G = g[g.ret > 0]; out["tous_signaux"][p] = dict(n=len(g), win=round(len(G) / len(g) * 100, 1), esp=round(g.ret.mean(), 2), mg=round(G.ret.mean(), 2), mp=round(g[g.ret <= 0].ret.mean(), 2), sorties={k: int(v) for k, v in g.why.value_counts().items()})
for p in ("P1", "P2"):
    E0 = E[(E.t < COUPURE) if p == "P1" else (E.t >= COUPURE)].reset_index(drop=True); g = P[P.per == p]
    out["portefeuille"][p] = dict(resume(E0), trades=len(g), win=round((g.ret > 0).mean() * 100, 1), esp=round(g.ret.mean(), 2))
out["portefeuille"]["total"] = dict(resume(E), trades=len(P))
json.dump(out, open(f"{D}/robot_swing_{TRI}_{MAXP}.json", "w"), indent=1); print(TRI, MAXP, json.dumps(out["portefeuille"]))
