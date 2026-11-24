"""Scenarios « publication de resultats » sur 10 ans (barres journalieres ajustees) :
jour de reaction = jour de l'annonce si l'ecart d'ouverture y est le plus fort (annonce avant l'ouverture), sinon le lendemain (apres la cloture).
Entree 1 : a l'ouverture du jour de reaction, revente a la cloture. Entree 2 (swing) : a la cloture du jour de reaction, revente 3 et 5 seances plus tard.
Comparaison : les memes ecarts d'ouverture les jours sans resultats. Apprentissage 2016-2022, controle 2023-2026."""
import os, glob, numpy as np, pandas as pd, json
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
E = pd.read_parquet(f"{D}/resultats.parquet").dropna(subset=["sym"]); E = E.drop_duplicates(["sym", "d"])
ev = {s: g.d.tolist() for s, g in E.groupby("sym")}; surp = {(r.sym, r.d): r.surp for r in E.itertuples()}
rows, autres = [], []
for p in sorted(glob.glob(f"{D}/data1d/*.npz")):
    sym = os.path.basename(p)[:-4]
    try: A = np.load(p)["a"]
    except Exception: continue
    if len(A) < 60: continue
    t, o, h, l, c, v = A.T
    d = pd.to_datetime(t, unit="s").strftime("%Y-%m-%d").values
    dv20 = pd.Series(c * v).rolling(20).mean().shift(1).values
    idx = {x: i for i, x in enumerate(d)}
    marque = np.zeros(len(d), bool)
    for de in ev.get(sym, []):
        i = np.searchsorted(d, de)
        if i <= 0 or i + 6 >= len(d): continue
        gA, gB = o[i] / c[i - 1] - 1, o[i + 1] / c[i] - 1
        R = i if (d[i] == de and abs(gA) >= abs(gB)) else i + 1
        marque[max(0, i - 1):i + 3] = True
        if c[R - 1] < 1 or not (dv20[R] >= 1e6): continue
        rows.append({"sym": sym, "d": d[R], "an": int(d[R][:4]), "gap": (o[R] / c[R - 1] - 1) * 100, "surp": surp.get((sym, de)), "pc": c[R - 1],
                     "rc": (c[R] / o[R] - 1) * 100, "r3": (c[R + 3] / c[R] - 1) * 100, "r5": (c[R + 5] / c[R] - 1) * 100, "avant": R == i})
    # jours sans resultats (comparaison), echantillon
    for i in range(21, len(d) - 6, 3):
        if marque[i] or c[i - 1] < 1 or not (dv20[i] >= 1e6): continue
        autres.append((d[i], int(d[i][:4]), (o[i] / c[i - 1] - 1) * 100, (c[i] / o[i] - 1) * 100, (c[i + 5] / c[i] - 1) * 100))
df = pd.DataFrame(rows); df = df[df.gap.between(-60, 300)]
B = pd.DataFrame(autres, columns=["d", "an", "gap", "rc", "r5"]); B = B[B.gap.between(-60, 300)]
df["per"] = np.where(df.an <= 2022, "A", "T"); B["per"] = np.where(B.an <= 2022, "A", "T")
df.to_parquet(f"{D}/resu_ev.parquet")
print(len(df), "publications avec reaction mesurable ;", round(df.avant.mean() * 100), "% avant l'ouverture ; surprise connue", round(df.surp.notna().mean() * 100), "%")
GAPS = [(-100, -10, "≤ −10 %"), (-10, -5, "−10 à −5 %"), (-5, -2, "−5 à −2 %"), (-2, 2, "−2 à +2 %"), (2, 5, "+2 à +5 %"), (5, 10, "+5 à +10 %"), (10, 1000, "≥ +10 %")]
SU = [(None, ""), ((10, 1e9), " · bénéfice au-dessus des attentes de 10 % ou plus"), ((0, 10), " · bénéfice au-dessus des attentes (0 à 10 %)"), ((-1e9, 0), " · bénéfice sous les attentes")]
def cl(v, g):
    n = len(v); m = v.mean(); e = (v - m).groupby(g.values).sum(); return m, (np.sqrt((e ** 2).sum()) / n if n > 1 else np.nan)
out = []
for a, b, gl in GAPS:
    for su, sl in SU:
        m = df.gap.between(a, b, inclusive="left") & ((df.surp >= su[0]) & (df.surp < su[1]) if su else True)
        mb = B.gap.between(a, b, inclusive="left")
        for ent, col, colb in (("o", "rc", "rc"), ("s5", "r5", "r5")):
            r = {"lib": f"résultats publiés · écart d'ouverture {gl}{sl}", "ent": ent}
            for per in ("A", "T"):
                s = df[m & (df.per == per)]; sb = B[mb & (B.per == per)]
                if len(s) < 30: break
                up, se = cl((s[col] > 0).astype(float), s.d); mo, sem = cl(s[col].clip(-50, 50), s.d)
                ball = B[B.per == per]  # reference : toutes les seances sans resultats (meme horizon), quel que soit l'ecart
                r[per] = dict(n=len(s), p=up, se=se, m=mo, sem=sem, ref=float((ball[colb] > 0).mean()), refg=float((sb[colb] > 0).mean()), gm=float(s[col][s[col] > 0].mean()), pm=float(s[col][s[col] <= 0].mean()))
            else:
                out.append(r)
res = []
for r in out:
    A, T = r["A"], r["T"]; n = A["n"] + T["n"]
    zA, zT = (A["p"] - A["ref"]) / A["se"], (T["p"] - T["ref"]) / T["se"]; zmA, zmT = A["m"] / A["sem"], T["m"] / T["sem"]
    force, sens = "non", None
    for za, zt in ((zA, zT), (zmA, zmT)):
        if np.sign(za) == np.sign(zt) and min(abs(za), abs(zt)) >= 2: force, sens = "confirme", "haut" if za > 0 else "bas"; break
    if force == "non":
        for za, zt in ((zA, zT), (zmA, zmT)):
            if np.sign(za) == np.sign(zt) and min(abs(za), abs(zt)) >= 1: force, sens = "indice", "haut" if za > 0 else "bas"; break
    if sens and any(z * (1 if sens == "haut" else -1) <= -1 for z in (zA, zT, zmA, zmT)): force = "instable"
    w = lambda k: (A[k] * A["n"] + T[k] * T["n"]) / n
    res.append(dict(lib=r["lib"], ent=r["ent"], n=n, p=w("p") * 100, ref=w("ref") * 100, refg=w("refg") * 100, m=w("m"), gm=w("gm"), pm=w("pm"), pA=A["p"] * 100, pT=T["p"] * 100, force=force, sens=sens,
                    se=np.sqrt((A["se"] * A["n"]) ** 2 + (T["se"] * T["n"]) ** 2) / n * 100))
R = pd.DataFrame(res); R.to_csv(f"{D}/resu_scen.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(R.round(2).to_string())
X = R.dropna(subset=["pA", "pT"]); print("concordance des deux periodes (ecart a la reference) : correlation", round(float(np.corrcoef(X.pA - X.ref, X.pT - X.ref)[0, 1]), 2))
