"""Rejeu des scenarios sur jours5m.parquet : pour chaque scenario, resultats sur la periode d'apprentissage (2/3 des jours)
et sur la periode de controle (1/3, jamais vue), avec erreurs-types groupees par jour (les actions bougent ensemble).
Sortie : scenarios.csv + resume a l'ecran.
"""
import os
import numpy as np, pandas as pd, json, sys
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(f"{D}/jours5m.parquet")
U = json.load(open(f"{D}/univers.json"))
# filtres de base : cours >= 1 $, liquidite >= 1 M$/jour, au moins 5 seances d'historique ; gaps absurdes (divisions d'actions) ecartes
df = df[(df.prevC >= 1) & (df.dv10 >= 1e6) & df.gap.between(-60, 300)].copy()
jours = sorted(df.d.unique()); cut = jours[int(len(jours) * 2 / 3)]
df["per"] = np.where(df.d < cut, "A", "T")
mk = df[df.sym.isin(["SPY", "QQQ"])].pivot_table(index="d", columns="sym", values=["gap", "rc_o"])
mk.columns = [a + "_" + b for a, b in mk.columns]; df = df.join(mk, on="d")
df["g925"] = (df.p925 / df.prevC - 1) * 100; df["g800"] = (df.p800 / df.prevC - 1) * 100
df["tr_pre"] = (df.p925 / df.p830 - 1) * 100  # tendance de la derniere heure de pre-marche
df["f15"] = (df.p945 / df.O - 1) * 100  # 15 premieres minutes
df["posOR"] = (df.p945 - df.or15L) / (df.or15H - df.or15L).replace(0, np.nan)
BIG = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "META", "AMZN", "GOOGL", "PLTR", "MSTR", "COIN", "SMCI", "AVGO", "NFLX", "MU", "INTC", "TSM", "ORCL", "UBER", "HOOD"]
df["big"] = df.sym.isin(BIG)
LIQ = df.dv10 >= 1e9  # entrees en pre-marche : seulement les actions tres echangees (ailleurs, le dernier prix du pre-marche n'est pas un prix qu'on aurait obtenu)

def stats(sub, ent, x):
    """sub : lignes du scenario ; ent : entree ; x : objectif/stop en %"""
    w = sub[f"w{x}_{ent}"]; rc = sub[f"rc_{ent}"]
    ok = rc.notna()
    sub, w, rc = sub[ok], w[ok], rc[ok]
    n = len(sub)
    if n == 0: return None
    up = (rc > 0).astype(float); hit = (w == 1).astype(float); stop = (w == 0).astype(float)
    def cl(v):  # moyenne et erreur-type groupee par jour
        m = v.mean(); e = (v - m).groupby(sub.d).sum(); se = np.sqrt((e ** 2).sum()) / n if n > 1 else np.nan
        return m, se
    r = {"n": n, "jours": sub.d.nunique(), "actions": sub.sym.nunique()}
    r["p_up"], r["se_up"] = cl(up); r["p_hit"], r["se_hit"] = cl(hit); r["p_stop"] = stop.mean()
    r["moy"], r["se_moy"] = cl(rc.clip(-50, 50)); r["med"] = rc.median()
    r["gain_moy"] = rc[rc > 0].mean(); r["perte_moy"] = rc[rc <= 0].mean()
    return r

S = []  # (famille, entree, libelle, masque)
def add(fam, ent, lib, m): S.append((fam, ent, lib, m))
GAPS = [(-100, -10, "≤ −10 %"), (-10, -5, "−10 à −5 %"), (-5, -2, "−5 à −2 %"), (-2, 2, "−2 à +2 %"), (2, 5, "+2 à +5 %"), (5, 10, "+5 à +10 %"), (10, 20, "+10 à +20 %"), (20, 1000, "≥ +20 %")]
for a, b, gl in GAPS:
    g = df.gap.between(a, b, inclusive="left")
    add("A ouverture", "o", f"écart d'ouverture {gl}", g)
    add("A ouverture", "o", f"écart {gl} · pré-marché en hausse sur la dernière heure", g & (df.tr_pre > 0.5))
    add("A ouverture", "o", f"écart {gl} · pré-marché en baisse sur la dernière heure", g & (df.tr_pre < -0.5))
    add("A ouverture", "o", f"écart {gl} · Nasdaq (QQQ) en hausse à l'ouverture", g & (df.gap_QQQ > 0.3))
    add("A ouverture", "o", f"écart {gl} · Nasdaq (QQQ) en baisse à l'ouverture", g & (df.gap_QQQ < -0.3))
    add("A ouverture", "o", f"écart {gl} · veille en forte hausse (≥ +5 %)", g & (df.r1j >= 5))
    add("A ouverture", "o", f"écart {gl} · veille en forte baisse (≤ −5 %)", g & (df.r1j <= -5))
    add("A ouverture", "o", f"écart {gl} · action à moins de 5 $", g & (df.prevC < 5))
    add("A ouverture", "o", f"écart {gl} · action à 20 $ et plus", g & (df.prevC >= 20))
    # 9h45 : apres les 15 premieres minutes
    for f, fl in ((df.f15 >= 1, "15 premières min en hausse (≥ +1 %)"), (df.f15 <= -1, "15 premières min en baisse (≤ −1 %)"), (df.f15.abs() < 1, "15 premières min stables")):
        add("B 9h45", "945", f"écart {gl} · {fl}", g & f)
        add("B 9h45", "945", f"écart {gl} · {fl} · volume d'ouverture ≥ 3× la normale", g & f & (df.rv15 >= 3))
    add("B 9h45", "945", f"écart {gl} · cours en haut de la fourchette des 15 premières min", g & (df.posOR >= 0.8))
    add("B 9h45", "945", f"écart {gl} · cours en bas de la fourchette des 15 premières min", g & (df.posOR <= 0.2))
    # 9h25 : achat en pre-marche
    g9 = LIQ & df.g925.between(a, b, inclusive="left")
    add("C pré-marché 9h25 · actions très échangées", "925", f"hausse en pré-marché {gl} à 9h25", g9)
    add("C pré-marché 9h25 · actions très échangées", "925", f"pré-marché {gl} à 9h25 · dernière heure en hausse", g9 & (df.tr_pre > 0.5))
    add("C pré-marché 9h25 · actions très échangées", "925", f"pré-marché {gl} à 9h25 · dernière heure en baisse", g9 & (df.tr_pre < -0.5))
for a, b, gl in [(5, 10, "+5 à +10 %"), (10, 20, "+10 à +20 %"), (20, 50, "+20 à +50 %"), (50, 1000, "≥ +50 %"), (-100, -5, "≤ −5 %")]:
    g8 = LIQ & df.g800.between(a, b, inclusive="left")
    add("D pré-marché 8h · actions très échangées", "800", f"déjà {gl} à 8h (heure de New York, 14h Paris)", g8)
for S_ in (5, 10, 20):
    has = df[f"dem{S_}_m"].notna()
    add("E démarrage en séance", f"dem{S_}", f"démarrage : dépasse +{S_} % avec une barre de volume ≥ 3×", has)
    for t0, t1, tl in ((585, 660, "entre 9h45 et 11h"), (660, 780, "entre 11h et 13h"), (780, 900, "entre 13h et 15h")):
        add("E démarrage en séance", f"dem{S_}", f"démarrage +{S_} % {tl} (New York)", has & df[f"dem{S_}_m"].between(t0, t1, inclusive="left"))
    add("E démarrage en séance", f"dem{S_}", f"démarrage +{S_} % · barre de volume ≥ 10×", has & (df[f"dem{S_}_x"] >= 10))
    add("E démarrage en séance", f"dem{S_}", f"démarrage +{S_} % · ouverture déjà en hausse (écart ≥ +{S_/2:g} %)", has & (df.gap >= S_ / 2))
    add("E démarrage en séance", f"dem{S_}", f"démarrage +{S_} % · parti de l'ouverture (écart < +2 %)", has & (df.gap < 2))
    add("E démarrage en séance", f"dem{S_}", f"démarrage +{S_} % · action à moins de 5 $", has & (df.prevC < 5))
# grandes valeurs des ETF x3 (objectifs petits : ±1 % sur l'action ≈ ±3 % sur l'ETF)
GB = [(-100, -2, "≤ −2 %"), (-2, -0.5, "−2 à −0,5 %"), (-0.5, 0.5, "−0,5 à +0,5 %"), (0.5, 2, "+0,5 à +2 %"), (2, 100, "≥ +2 %")]
for a, b, gl in GB:
    g = df.big & df.gap.between(a, b, inclusive="left")
    add("F grandes valeurs (ETF ×3)", "o", f"écart {gl}", g)
    add("F grandes valeurs (ETF ×3)", "o", f"écart {gl} · Nasdaq en hausse", g & (df.gap_QQQ > 0.3))
    add("F grandes valeurs (ETF ×3)", "o", f"écart {gl} · Nasdaq en baisse", g & (df.gap_QQQ < -0.3))
    for f, fl in ((df.f15 >= 0.5, "15 premières min en hausse"), (df.f15 <= -0.5, "15 premières min en baisse")):
        add("F grandes valeurs (ETF ×3)", "945", f"écart {gl} · {fl}", g & f)
    g9 = df.big & df.g925.between(a, b, inclusive="left")
    add("F grandes valeurs (ETF ×3)", "925", f"pré-marché {gl} à 9h25", g9)

CIB = {"o": 3, "945": 3, "925": 3, "800": 5, "dem5": 3, "dem10": 5, "dem20": 5}
def cible(fam, ent): return 1 if fam.startswith("F") else CIB[ent]

rows = []
for fam, ent, lib, m in S:
    x = cible(fam, ent)
    base = df[df[f"rc_{ent}"].notna() & (df.big if fam.startswith("F") else LIQ if ent == "925" else True)] if ent in ("o", "945", "925") else None
    for per in ("A", "T"):
        sub = df[m.fillna(False) & (df.per == per)]
        r = stats(sub, ent, x)
        if r is None: continue
        if ent in ("o", "945", "925"):
            b = stats(base[base.per == per], ent, x); r["base_up"], r["base_hit"], r["base_moy"] = b["p_up"], b["p_hit"], b["moy"]
        rows.append(dict(fam=fam, ent=ent, lib=lib, cible=x, per=per, **r))
R = pd.DataFrame(rows)
R.to_csv(f"{D}/scenarios.csv", index=False)
print("periode A :", jours[0], "->", cut, " periode T :", cut, "->", jours[-1], " lignes", len(df))
