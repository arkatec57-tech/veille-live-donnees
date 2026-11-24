import os
import pandas as pd, numpy as np
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(f"{D}/quot.parquet")
df["per"] = np.where(df.an <= 2022, "A", "T")
spy = df[df.sym == "QQQ"].set_index("d").gap.rename("gapQ"); df = df.join(spy, on="d")
GAPS = [(-100, -10, "≤ −10 %"), (-10, -5, "−10 à −5 %"), (-5, -2, "−5 à −2 %"), (-2, 2, "−2 à +2 %"), (2, 5, "+2 à +5 %"), (5, 10, "+5 à +10 %"), (10, 20, "+10 à +20 %"), (20, 1000, "≥ +20 %")]
def cl(v, g):
    n = len(v); m = v.mean(); e = (v - m).groupby(g).sum(); return m, np.sqrt((e ** 2).sum()) / n
BASE = {p: float((df[df.per == p].rc > 0).mean()) for p in ("A", "T")}; BASE["*"] = float((df.rc > 0).mean())
rows = []
for a, b, gl in GAPS:
    g = df.gap.between(a, b, inclusive="left")
    for cond, cl_ in ((g, ""), (g & (df.pc < 5), " · moins de 5 $"), (g & (df.pc >= 20), " · 20 $ et plus"), (g & (df.r1j >= 5), " · veille ≥ +5 %"), (g & (df.r1j <= -5), " · veille ≤ −5 %"),
                      (g & (df.gapQ > 0.3), " · Nasdaq en hausse"), (g & (df.gapQ < -0.3), " · Nasdaq en baisse")):
        for per in ("A", "T", "*"):
            s = df[cond & ((df.per == per) if per != "*" else True)]
            if len(s) < 30: continue
            up, se = cl((s.rc > 0).astype(float), s.d); mo, sem = cl(s.rc.clip(-50, 50), s.d)
            rows.append(dict(lib="écart " + gl + cl_, per=per, n=len(s), p_up=up, se_up=se, moy=mo, se_moy=sem, hit3=s.hit3.mean(), low3=s.low3.mean(), rn=s.rn.mean(),
                             gain=s.rc[s.rc > 0].mean(), perte=s.rc[s.rc <= 0].mean(), base_up=BASE[per]))
R = pd.DataFrame(rows); R.to_csv(f"{D}/quot_scen.csv", index=False)
base = df.groupby("per").apply(lambda s: (s.rc > 0).mean()); print("base p_up", base.round(3).to_dict(), "n", len(df))
pd.set_option("display.width", 250); print(R[R.per != "*"].round(3).to_string())
# stabilite par annee pour l'ecart d'ouverture
for a, b, gl in GAPS:
    g = df[df.gap.between(a, b, inclusive="left")]
    print(gl, g.groupby("an").apply(lambda s: f"{(s.rc > 0).mean() * 100:.0f}%({len(s)})").to_dict())
# etalonnage sur 10 ans : la probabilite 2016-2022 annonce-t-elle 2023-2026 ?
P = R[R.per != "*"].pivot_table(index="lib", columns="per", values=["p_up", "n"])
P = P[(P[("n", "A")] >= 300) & (P[("n", "T")] >= 100)]
pa, pt = P[("p_up", "A")], P[("p_up", "T")]
print("10 ans : correlation A/T", round(float(np.corrcoef(pa, pt)[0, 1]), 2), "scenarios", len(P), "erreur moyenne", round(float((pa - pt).abs().mean() * 100), 1), "points")
B = pd.cut(pa, [0, .35, .45, .48, .52, .55, .65, 1])
print(pd.DataFrame({"annonce": pa, "constate": pt}).groupby(B, observed=True).agg(["mean", "count"]).round(3))
