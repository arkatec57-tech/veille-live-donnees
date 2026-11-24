"""Rejeu des scenarios sur ~3 ans de barres d'une heure. Periode A : avant le 1er juillet 2025 ; periode T : ensuite (jamais vue).
Probabilites : cours a la cloture au-dessus du prix d'achat ; objectif +x avant stop -x (cas tranches) ; resultat moyen."""
import os
import numpy as np, pandas as pd
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(f"{D}/jours1h.parquet")
df = df[(df.prevC >= 1) & (df.dv10 >= 1e6) & df.gap.between(-60, 300)].copy()
COUP = "2025-07-01"; df["per"] = np.where(df.d < COUP, "A", "T")
mk = df[df.sym.isin(["QQQ"])].set_index("d")[["gap", "rc_o"]].add_suffix("_QQQ"); df = df.join(mk, on="d")
df["g930"] = (df.p930 / df.prevC - 1) * 100; df["g800"] = (df.p800 / df.prevC - 1) * 100
df["tr_pre"] = (df.p930 / df.p900 - 1) * 100; df["f1h"] = (df.p1030 / df.O - 1) * 100
BIG = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "META", "AMZN", "GOOGL", "PLTR", "MSTR", "COIN", "SMCI", "AVGO", "NFLX", "MU", "INTC", "TSM", "ORCL", "UBER", "HOOD"]
df["big"] = df.sym.isin(BIG)
LIQ = df.dv10 >= 1e9  # entrees en pre-marche : seulement les actions tres echangees (ailleurs, le dernier prix du pre-marche n'est pas un prix qu'on aurait obtenu)

def stats(sub, ent, x):
    rc = sub[f"rc_{ent}"]; ok = rc.notna(); sub = sub[ok]; rc = rc[ok]; n = len(sub)
    if n == 0: return None
    w = sub[f"w{x}_{ent}"]
    up = (rc > 0).astype(float); hit = (w == 1).astype(float); stop = (w == 0).astype(float)
    def cl(v):
        m = v.mean(); e = (v - m).groupby(sub.d).sum(); return m, (np.sqrt((e ** 2).sum()) / n if n > 1 else np.nan)
    r = {"n": n, "jours": sub.d.nunique(), "actions": sub.sym.nunique()}
    r["p_up"], r["se_up"] = cl(up); r["p_hit"], r["se_hit"] = cl(hit); r["p_stop"] = stop.mean(); r["p_amb"] = w.isna().mean()
    r["moy"], r["se_moy"] = cl(rc.clip(-50, 50)); r["med"] = rc.median(); r["gain_moy"] = rc[rc > 0].mean(); r["perte_moy"] = rc[rc <= 0].mean()
    return r

S = []
def add(fam, ent, lib, m): S.append((fam, ent, lib, m))
GAPS = [(-100, -10, "≤ −10 %"), (-10, -5, "−10 à −5 %"), (-5, -2, "−5 à −2 %"), (-2, 2, "−2 à +2 %"), (2, 5, "+2 à +5 %"), (5, 10, "+5 à +10 %"), (10, 20, "+10 à +20 %"), (20, 1000, "≥ +20 %")]
for a, b, gl in GAPS:
    g = df.gap.between(a, b, inclusive="left")
    add("A ouverture", "o", f"écart d'ouverture {gl}", g)
    add("A ouverture", "o", f"écart {gl} · pré-marché en hausse sur la dernière demi-heure", g & (df.tr_pre > 0.5))
    add("A ouverture", "o", f"écart {gl} · pré-marché en baisse sur la dernière demi-heure", g & (df.tr_pre < -0.5))
    add("A ouverture", "o", f"écart {gl} · Nasdaq (QQQ) en hausse à l'ouverture", g & (df.gap_QQQ > 0.3))
    add("A ouverture", "o", f"écart {gl} · Nasdaq (QQQ) en baisse à l'ouverture", g & (df.gap_QQQ < -0.3))
    add("A ouverture", "o", f"écart {gl} · veille en forte hausse (≥ +5 %)", g & (df.r1j >= 5))
    add("A ouverture", "o", f"écart {gl} · veille en forte baisse (≤ −5 %)", g & (df.r1j <= -5))
    add("A ouverture", "o", f"écart {gl} · 5 jours en forte hausse (≥ +15 %)", g & (df.r5j >= 15))
    add("A ouverture", "o", f"écart {gl} · 5 jours en forte baisse (≤ −15 %)", g & (df.r5j <= -15))
    add("A ouverture", "o", f"écart {gl} · action à moins de 5 $", g & (df.prevC < 5))
    add("A ouverture", "o", f"écart {gl} · action à 20 $ et plus", g & (df.prevC >= 20))
    for f, fl in ((df.f1h >= 1, "1re heure en hausse (≥ +1 %)"), (df.f1h <= -1, "1re heure en baisse (≤ −1 %)"), (df.f1h.abs() < 1, "1re heure stable")):
        add("B après la 1re heure (10h30)", "1030", f"écart {gl} · {fl}", g & f)
        add("B après la 1re heure (10h30)", "1030", f"écart {gl} · {fl} · volume de la 1re heure ≥ 3× la normale", g & f & (df.rv1h >= 3))
    g9 = LIQ & df.g930.between(a, b, inclusive="left")
    add("C pré-marché (fin, 9h30) · actions très échangées", "930", f"pré-marché {gl} juste avant l'ouverture", g9)
    add("C pré-marché (fin, 9h30) · actions très échangées", "930", f"pré-marché {gl} · dernière demi-heure en hausse", g9 & (df.tr_pre > 0.5))
    add("C pré-marché (fin, 9h30) · actions très échangées", "930", f"pré-marché {gl} · dernière demi-heure en baisse", g9 & (df.tr_pre < -0.5))
for a, b, gl in [(5, 10, "+5 à +10 %"), (10, 20, "+10 à +20 %"), (20, 50, "+20 à +50 %"), (50, 1000, "≥ +50 %"), (-100, -5, "≤ −5 %")]:
    g8 = LIQ & df.g800.between(a, b, inclusive="left")
    add("D pré-marché tôt (8h New York) · actions très échangées", "800", f"déjà {gl} à 8h (14h Paris)", g8)
GB = [(-100, -2, "≤ −2 %"), (-2, -0.5, "−2 à −0,5 %"), (-0.5, 0.5, "−0,5 à +0,5 %"), (0.5, 2, "+0,5 à +2 %"), (2, 100, "≥ +2 %")]
for a, b, gl in GB:
    g = df.big & df.gap.between(a, b, inclusive="left")
    add("F grandes valeurs des ETF ×3", "o", f"écart {gl}", g)
    add("F grandes valeurs des ETF ×3", "o", f"écart {gl} · Nasdaq en hausse", g & (df.gap_QQQ > 0.3))
    add("F grandes valeurs des ETF ×3", "o", f"écart {gl} · Nasdaq en baisse", g & (df.gap_QQQ < -0.3))
    add("F grandes valeurs des ETF ×3", "o", f"écart {gl} · veille en forte baisse (≤ −3 %)", g & (df.r1j <= -3))
    add("F grandes valeurs des ETF ×3", "o", f"écart {gl} · veille en forte hausse (≥ +3 %)", g & (df.r1j >= 3))
    for f, fl in ((df.f1h >= 0.5, "1re heure en hausse"), (df.f1h <= -0.5, "1re heure en baisse")):
        add("F grandes valeurs des ETF ×3", "1030", f"écart {gl} · {fl}", g & f)
    g9 = df.big & df.g930.between(a, b, inclusive="left")
    add("F grandes valeurs des ETF ×3", "930", f"pré-marché {gl} juste avant l'ouverture", g9)
CIB = {"o": 3, "1030": 3, "930": 3, "800": 5}
rows = []
for fam, ent, lib, m in S:
    x = 1 if fam.startswith("F") else CIB[ent]
    base = df[df[f"rc_{ent}"].notna() & (df.big if fam.startswith("F") else LIQ if ent in ("930", "800") else True)]
    for per in ("A", "T"):
        sub = df[m.fillna(False) & (df.per == per)]
        r = stats(sub, ent, x)
        if r is None: continue
        b = stats(base[base.per == per], ent, x); r["base_up"], r["base_hit"], r["base_moy"] = b["p_up"], b["p_hit"], b["moy"]
        rows.append(dict(fam=fam, ent=ent, lib=lib, cible=x, per=per, **r))
R = pd.DataFrame(rows); R.to_csv(f"{D}/scenarios_h.csv", index=False)
print("A :", df[df.per == "A"].d.min(), "->", df[df.per == "A"].d.max(), " T :", df[df.per == "T"].d.min(), "->", df[df.per == "T"].d.max(), " lignes", len(df), " scenarios", len(S))
print("base p_up", df.groupby("per").apply(lambda s: (s.rc_o > 0).mean()).round(3).to_dict())
