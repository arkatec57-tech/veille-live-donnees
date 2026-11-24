"""Assemble les trois sources (10 ans journalier, 3 ans horaire, 60 jours en 5 min) en un tableau pour la page.
Sorties : page/scenarios.json ({rows, meta}) et page/etalonnage.json ({rows, meta})."""
import json, os, hashlib, numpy as np, pandas as pd
def sid(*a): return hashlib.md5("|".join(map(str, a)).encode()).hexdigest()[:8]
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__))); os.makedirs(f"{D}/page", exist_ok=True)
ENT = {"o": "à l'ouverture (9h30 New York)", "945": "à 9h45 New York (après 15 min)", "1030": "à 10h30 New York (après 1 h)", "925": "en pré-marché à 9h25 New York",
       "930": "en pré-marché juste avant l'ouverture", "800": "en pré-marché à 8h New York", "dem5": "dès le démarrage (barre suivante)", "dem10": "dès le démarrage (barre suivante)", "dem20": "dès le démarrage (barre suivante)"}
SRC = {"q": "10 ans · journalier", "h": "3 ans · barres d'1 h", "m": "60 jours · barres de 5 min"}
rows = []
def pente(pairs):
    X = np.array(pairs); return float(np.clip(np.polyfit(X[:, 0], X[:, 1], 1)[0], 0, 1)) if len(X) >= 10 else 0.0
K = {}
for code_, f in (("h", "scenarios_h.csv"), ("m", "scenarios.csv")):
    if not os.path.exists(f"{D}/{f}"): K[code_] = 0.0; continue  # mise a jour hebdomadaire : pas de donnees 5 min
    R_ = pd.read_csv(f"{D}/{f}"); X = []
    for _, g in R_.groupby(["fam", "ent", "lib"]):
        a_, t_ = g[g.per == "A"], g[g.per == "T"]
        if len(a_) and len(t_) and a_.n.iloc[0] >= 100 and t_.n.iloc[0] >= 50 and not np.isnan(a_.base_up.iloc[0]): X.append((a_.p_up.iloc[0] - a_.base_up.iloc[0], t_.p_up.iloc[0] - t_.base_up.iloc[0]))
    K[code_] = pente(X)
Q0 = pd.read_csv(f"{D}/quot_scen.csv"); P0 = Q0[Q0.per != "*"].pivot_table(index="lib", columns="per", values=["p_up", "n", "base_up"]); P0 = P0[(P0[("n", "A")] >= 300) & (P0[("n", "T")] >= 100)]
K["q"] = pente(list(zip(P0[("p_up", "A")] - P0[("base_up", "A")], P0[("p_up", "T")] - P0[("base_up", "T")])))
print("retrait", K)
def reel(code_, p, ref):  # probabilite realiste : on ne garde que la part de l'ecart qui s'est confirmee d'une periode a l'autre
    return None if p is None or ref is None or K[code_] == 0 else round(ref + K[code_] * (p - ref), 1)

def verdict(force, sens, ecart=None, moy=None):
    if force == "confirme" and ecart is not None and abs(ecart) < 3 and abs(moy or 0) < 0.3: force = "indice"  # vrai mais trop petit pour compter
    if force == "confirme": return "Favorable" if sens == "haut" else "Défavorable"
    if force == "indice": return "Léger avantage" if sens == "haut" else "Léger désavantage"
    if force == "instable": return "Instable"
    return "Pas d'avantage"

def r1(x, k=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)

for code, fn in (("h", "synthese_h.json"), ("m", "synthese.json")):
    if not os.path.exists(f"{D}/{fn}"): continue
    for o in json.load(open(f"{D}/{fn}"))["S"]:
        fam = o["fam"].split(" ", 1)[1] if o["fam"][1] == " " else o["fam"]
        rows.append({"id": code + "-" + sid(o["fam"], o["ent"], o["lib"]), "source": SRC[code], "famille": fam[0].upper() + fam[1:], "entree": ENT.get(o["ent"], o["ent"]), "scenario": o["lib"][0].upper() + o["lib"][1:],
                     "cas": o["n"], "jours": o["jours"], "p_up": r1(o["p_up"] * 100), "marge": r1(o["ic_up"] * 100), "ref": r1(o["ref_up"] * 100),
                     "cible": o["cible"], "p_cible": r1(o["p_dir"] * 100 if o.get("p_dir") is not None else None), "tranche": r1(o["dec"] * 100 if o.get("dec") is not None else None, 0),
                     "moy": r1(o["moy"], 2), "gain": r1(o.get("gain"), 2), "perte": r1(o.get("perte"), 2),
                     "p1": r1(o["A"]["p_up"] * 100), "p2": r1(o["T"]["p_up"] * 100), "verdict": verdict(o["force"], o["sens"], (o["p_up"] - o["ref_up"]) * 100, o["moy"]),
                     "p_reel": reel(code, o["p_up"] * 100, o["ref_up"] * 100)})
        if code == "m": rows[-1]["verdict"] = {"Favorable": "À confirmer (hausse)", "Défavorable": "À confirmer (baisse)", "Léger avantage": "Pas d'avantage", "Léger désavantage": "Pas d'avantage"}.get(rows[-1]["verdict"], rows[-1]["verdict"])
# 10 ans journalier : achat a l'ouverture, vente a la cloture ; verdict par la meme regle (deux periodes)
Q = pd.read_csv(f"{D}/quot_scen.csv")
for lib, g in Q.groupby("lib", sort=False):
    A, T, S = (g[g.per == p] for p in ("A", "T", "*"))
    if not (len(A) and len(T) and len(S)): continue
    A, T, S = A.iloc[0], T.iloc[0], S.iloc[0]
    if S.n < 100: continue
    zA, zT = (A.p_up - A.base_up) / A.se_up, (T.p_up - T.base_up) / T.se_up
    zmA, zmT = A.moy / A.se_moy, T.moy / T.se_moy
    force, sens = "non", None
    for za, zt in ((zA, zT), (zmA, zmT)):
        if np.sign(za) == np.sign(zt) and min(abs(za), abs(zt)) >= 2: force, sens = "confirme", ("haut" if za > 0 else "bas"); break
    if force == "non":
        for za, zt in ((zA, zT), (zmA, zmT)):
            if np.sign(za) == np.sign(zt) and min(abs(za), abs(zt)) >= 1: force, sens = "indice", ("haut" if za > 0 else "bas"); break
    if sens:
        s0 = 1 if sens == "haut" else -1
        if any(z * s0 <= -1 for z in (zA, zT, zmA, zmT)): force = "instable"
    rows.append({"id": "q-" + sid(lib), "source": SRC["q"], "famille": "Écart à l'ouverture (10 ans)", "entree": ENT["o"], "scenario": lib[0].upper() + lib[1:],
                 "cas": int(S.n), "jours": None, "p_up": r1(S.p_up * 100), "marge": r1(1.645 * S.se_up * 100), "ref": r1(S.base_up * 100),
                 "cible": 3, "p_cible": None, "tranche": None, "moy": r1(S.moy, 2), "gain": r1(S.gain, 2), "perte": r1(S.perte, 2),
                 "p1": r1(A.p_up * 100), "p2": r1(T.p_up * 100), "verdict": verdict(force, sens, (S.p_up - S.base_up) * 100, S.moy), "p_reel": reel("q", S.p_up * 100, S.base_up * 100),
                 "touche_haut": r1(S.hit3 * 100), "touche_bas": r1(S.low3 * 100)})
code = {k: open(f"{D}/{k}").read() for k in ("fetch.py", "jours.py", "heures.py", "quot.py", "scen.py", "scen_h.py", "quot_scen.py", "analyse.py", "export.py") if os.path.exists(f"{D}/{k}")}
meta = [{"label": "Données", "value": "Cours Yahoo Finance de 5 826 actions américaines à 1 $ et plus (liste Nasdaq du 23/11/2026) : journalier 2016-2026, barres d'1 h depuis déc. 2023, barres de 5 min sur 60 jours (pré-marché en prix seulement)."},
        {"label": "Règles", "value": "Pas de frais. Objectif et stop touchés dans la même barre : compté comme stop (5 min) ou non tranché (1 h). Deux périodes mesurées séparément ; « Favorable » ou « Défavorable » seulement si les deux vont nettement dans le même sens (écart d'au moins 2 erreurs-types, erreurs groupées par jour)."}]
meta.append({"label": "Probabilité réaliste", "value": "Référence + pente × (mesure − référence), la pente étant ce qui s'est confirmé d'une période à l'autre : " + ", ".join(f"{SRC[k]} {v:.2f}" for k, v in K.items()) + ". Sur 60 jours, rien ne s'est confirmé en moyenne : pas de probabilité réaliste."})
meta += [{"label": k, "value": v, "type": "code", "lang": "python"} for k, v in code.items()]
json.dump({"rows": rows, "meta": meta}, open(f"{D}/page/scenarios.json", "w"), ensure_ascii=False)
# etalonnage : annonce (1re periode) contre constate (2e periode), par source
E = []
for code_, src in (("q", None), ("h", "synthese_h.json"), ("m", "synthese.json")):
    if code_ == "q":
        P = Q[Q.per != "*"].pivot_table(index="lib", columns="per", values=["p_up", "n"]); P = P[(P[("n", "A")] >= 300) & (P[("n", "T")] >= 100)]
        pa, pt = P[("p_up", "A")], P[("p_up", "T")]
    else:
        if not os.path.exists(f"{D}/{src}"): continue
        S_ = [o for o in json.load(open(f"{D}/{src}"))["S"] if o["A"]["n"] >= 100 and o["T"]["n"] >= 50]
        pa, pt = pd.Series([o["A"]["p_up"] for o in S_]), pd.Series([o["T"]["p_up"] for o in S_])
    for x, y in zip(pa, pt): E.append({"source": SRC[code_], "annonce": round(float(x) * 100, 1), "constate": round(float(y) * 100, 1)})
json.dump({"rows": E, "meta": [{"label": "Lecture", "value": "Chaque point est un scénario : probabilité mesurée sur la 1re période (annoncée) et ce qui s'est passé sur la 2e (constatée). Sur la diagonale, l'annonce était juste."}]},
          open(f"{D}/page/etalonnage.json", "w"), ensure_ascii=False)
R = pd.DataFrame(rows); print(len(R), "lignes"); print(R.groupby(["source", "verdict"]).size())
