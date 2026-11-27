"""Rejeu des règles du robot day trading sur 3 ans de barres d'une heure (séance 9h30-16h00 New York).
Entrées après la 1re heure (10h30) ; sortie à la clôture, au stop (plus bas de la 1re heure) ou à l'objectif. Long seulement, frais exclus,
achat au prix + 0,05 %, vente au prix - 0,05 %. Deux périodes : P1 jusqu'au 31/05/2025, P2 ensuite."""
import os, glob, json, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); SLIP = 0.0005; COUP = pd.Timestamp("2025-06-01", tz="UTC").timestamp()
def jours(A):
    t = A[:, 0]; off = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).values.astype("datetime64[s]").astype(np.int64) - t.astype(np.int64)
    loc = t.astype(np.int64) + off; jour = loc // 86400; hm = (loc % 86400) // 60
    m = (hm >= 570) & (hm < 960); A2 = A[m]; j2 = jour[m]; h2 = hm[m]; out = []
    if not len(A2): return out
    cut = np.flatnonzero(np.diff(j2)) + 1; deb = np.r_[0, cut]; fin = np.r_[cut, len(j2)]
    for a, b in zip(deb, fin):
        if b - a < 6 or h2[a] != 570: continue
        g = A2[a:b]; out.append((int(j2[a]), g[0, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5]))
    return out
REG = {}
for gmin in (2, 3, 5):
    REG[f"gap{gmin}_1h_verte"] = gmin
    REG[f"gap{gmin}_cassure_1h"] = gmin
REG["gapbaisse3_rebond"] = -3; REG["temoin_10h30"] = None
def rejouer(sym, A, out):
    J = jours(A)
    if len(J) < 60: return
    vol1 = [j[6][0] for j in J]; closes = [j[5][-1] for j in J]; dvs = [float((j[5] * j[6]).sum()) for j in J]
    for i in range(21, len(J)):
        dd, t0, o, h, l, c, v = J[i]; pc = closes[i - 1]; dv20 = np.mean(dvs[i - 20:i]); rv1 = v[0] / max(1, np.median(vol1[i - 20:i]))
        if pc < 5 or dv20 < 20e6: continue
        gap = (o[0] / pc - 1) * 100; s20 = np.mean(closes[i - 20:i]); orh, orl = h[0], l[0]
        def trade(k, e, stp, k0):  # e : prix d'entree, k0 : index de la barre d'entree (achat a sa fin si k0=0 -> 10h30)
            e *= (1 + SLIP); R0 = e - stp
            if R0 <= 0: return
            obj = e + 2 * R0; ps = None
            for b in range(k0 + 1, len(c)):
                if l[b] <= stp: ps = min(o[b], stp); break
                if h[b] >= obj: ps = max(o[b], obj); break
            if ps is None: ps = c[-1]
            out.append((sym, k, t0, gap, rv1, (ps * (1 - SLIP) / e - 1) * 100, R0 / e * 100))
        for k, g0 in REG.items():
            if k == "temoin_10h30":
                if i % 5 == 0: trade(k, c[0], c[0] * 0.98, 0)
            elif k == "gapbaisse3_rebond":
                if gap <= -3 and c[0] > o[0] and pc > s20: trade(k, c[0], orl, 0)
            elif k.endswith("1h_verte"):
                if gap >= g0 and c[0] > o[0] and (c[0] - l[0]) >= 0.5 * (h[0] - l[0]) and rv1 >= 1.5: trade(k, c[0], orl, 0)
            elif k.endswith("cassure_1h"):
                if gap >= g0 and rv1 >= 1.5:
                    for b in range(1, len(c) - 1):  # premiere barre qui depasse le plus haut de la 1re heure (pas apres 14h30)
                        if h[b] > orh:
                            trade(k, max(o[b], orh * 1.001), orl, b - 1 if o[b] > orh else b - 1); break
def main():
    T = []
    F = sorted(glob.glob(f"{D}/data1h/*.npz"))
    for n, f in enumerate(F):
        try: rejouer(os.path.basename(f)[:-4], np.load(f)["a"], T)
        except Exception as e: pass
        if n % 1000 == 0: print(n, len(T), flush=True)
    df = pd.DataFrame(T, columns=["sym", "k", "t", "gap", "rv1", "ret", "risque"]); df["per"] = np.where(df.t < COUP, "P1", "P2"); df.to_parquet(f"{D}/bt_day.parquet")
    res = {}
    for (k, p), g in df.groupby(["k", "per"]):
        G = g[g.ret > 0]; P0 = g[g.ret <= 0]
        res.setdefault(k, {})[p] = dict(n=len(g), win=round(len(G) / len(g) * 100, 1), esp=round(g.ret.mean(), 3), med=round(g.ret.median(), 3), mg=round(G.ret.mean(), 2) if len(G) else None, mp=round(P0.ret.mean(), 2) if len(P0) else None, pf=round(G.ret.sum() / max(1e-9, -P0.ret.sum()), 2))
    json.dump(res, open(f"{D}/bt_day.json", "w"), indent=1)
    for k, v in res.items(): print(f"{k:22s}", v.get("P1"), "|", v.get("P2"))
if __name__ == "__main__": main()
