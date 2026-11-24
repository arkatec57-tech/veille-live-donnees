"""Genere app/scenarios.js pour Veille Live : les scenarios d'entree qui ont tenu sur 10 ans ou 3 ans
(verdict Favorable, Defavorable, ou leger), avec des conditions structurees lisibles par l'application."""
import os
import json, re
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(f"{D}/page/scenarios.json"))["rows"]
GL = {"≤ −10 %": [-100, -10], "−10 à −5 %": [-10, -5], "−5 à −2 %": [-5, -2], "−2 à +2 %": [-2, 2], "+2 à +5 %": [2, 5], "+5 à +10 %": [5, 10],
      "+10 à +20 %": [10, 20], "≥ +20 %": [20, 1000], "+20 à +50 %": [20, 50], "≥ +50 %": [50, 1000], "≤ −5 %": [-100, -5]}
COND = {"pré-marché en hausse sur la dernière demi-heure": "pre+", "pré-marché en baisse sur la dernière demi-heure": "pre-",
        "dernière demi-heure en hausse": "pre+", "dernière demi-heure en baisse": "pre-",
        "nasdaq (qqq) en hausse à l'ouverture": "qqq+", "nasdaq (qqq) en baisse à l'ouverture": "qqq-", "nasdaq en hausse": "qqq+", "nasdaq en baisse": "qqq-",
        "veille en forte hausse (≥ +5 %)": "v+", "veille en forte baisse (≤ −5 %)": "v-", "veille ≥ +5 %": "v+", "veille ≤ −5 %": "v-",
        "5 jours en forte hausse (≥ +15 %)": "5j+", "5 jours en forte baisse (≤ −15 %)": "5j-",
        "action à moins de 5 $": "p<5", "moins de 5 $": "p<5", "action à 20 $ et plus": "p20", "20 $ et plus": "p20",
        "1re heure en hausse (≥ +1 %)": "h+", "1re heure en baisse (≤ −1 %)": "h-", "1re heure stable": "h=", "volume de la 1re heure ≥ 3× la normale": "rv3"}
VC = {"Favorable": "F", "Défavorable": "D", "Léger avantage": "f", "Léger désavantage": "d"}
out, rate = [], []
for r in R:
    if r["verdict"] not in VC or r["source"].startswith("60"): continue
    fam = r["famille"].lower()
    if fam.startswith("grandes valeurs"): continue
    lib = r["scenario"][0].lower() + r["scenario"][1:]
    if fam.startswith("ouverture") or fam.startswith("écart à l'ouverture"): e = "o"
    elif fam.startswith("après la 1re heure"): e = "1030"
    elif fam.startswith("pré-marché (fin"): e = "930"
    elif fam.startswith("pré-marché tôt"): e = "800"
    else: rate.append(lib); continue
    m = re.match(r"(?:écart d'ouverture |écart |pré-marché |déjà )(.+?)(?: juste avant l'ouverture| à 8h \(14h Paris\))?(?: · (.*))?$", lib)
    if not m or m.group(1) not in GL: rate.append(lib); continue
    cs = [c for c in (m.group(2) or "").split(" · ") if c]
    try: cc = [COND[c.lower()] for c in cs]
    except KeyError: rate.append(lib); continue
    out.append({"e": e, "g": GL[m.group(1)], "c": cc, "s": "q" if r["source"].startswith("10") else "h", "n": r["cas"], "p": r["p_up"], "r": r["p_reel"], "ref": r["ref"],
                "m": r["moy"], "gm": r["gain"], "pm": r["perte"], "v": VC[r["verdict"]], "lib": r["scenario"]})
# v2.06 : jours de publication de resultats (10 ans, calendrier Nasdaq) : ouverture -> cloture ("r") et swing 5 seances apres la cloture ("r5")
import pandas as pd
RS = pd.read_csv(f"{D}/resu_scen.csv")
SUC = {"bénéfice au-dessus des attentes de 10 % ou plus": "b10", "bénéfice au-dessus des attentes (0 à 10 %)": "b0", "bénéfice sous les attentes": "m"}
GLR = dict(GL); GLR["≥ +10 %"] = [10, 1000]
nr = 0
for r in RS.itertuples():
    m = re.match(r"résultats publiés · écart d'ouverture (.+?)(?: · (.*))?$", r.lib)
    if not m: rate.append(r.lib); continue
    ec = r.p - r.ref
    v = "N"
    if r.force == "confirme": v = ("F" if r.sens == "haut" else "D") if (abs(ec) >= 3 or abs(r.m) >= 0.3) else ("f" if r.sens == "haut" else "d")
    elif r.force == "indice": v = "f" if r.sens == "haut" else "d"
    out.append({"e": "r" if r.ent == "o" else "r5", "g": GLR[m.group(1)], "c": [], "su": SUC.get(m.group(2)) if m.group(2) else None, "s": "q", "n": int(r.n), "p": round(r.p, 1),
                "r": round(r.ref + 0.72 * (r.p - r.ref), 1), "ref": round(r.ref, 1), "refg": round(r.refg, 1), "m": round(r.m, 2), "gm": round(r.gm, 1), "pm": round(r.pm, 1), "v": v,
                "lib": "Résultats publiés · écart d'ouverture " + m.group(1) + (" · " + m.group(2) if m.group(2) else "")}); nr += 1
print(nr, "scenarios de resultats")
import datetime, glob as _g
info = {"date": os.environ.get("VL_DATE_ETUDE", datetime.date.today().isoformat()), "actions": len(_g.glob(f"{D}/data1d/*.npz")) or 5826, "note": "10 ans (journalier) et 3 ans (barres d'1 h), plus 10 ans de publications de résultats"}
js = "// v2.05 : scenarios d'entree mesures sur l'historique (genere par gen_app.py a partir de l'etude du 23/11/2026 ; ne pas modifier a la main)\n"
js += "window.VL = window.VL || {};\nwindow.VL.SCEN = " + json.dumps({"info": info, "L": out}, ensure_ascii=False, separators=(",", ":")) + ";\n"
open(os.environ.get("VL_SORTIE_JS", f"{D}/scenarios.js"), "w", encoding="utf-8").write(js)
json.dump({"info": info, "L": out}, open(os.environ.get("VL_SORTIE_JSON", f"{D}/scenarios.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))  # v2.06 : fichier lu par l'application chaque semaine
print(len(out), "scenarios retenus ;", len(rate), "non lus :", rate[:10])
from collections import Counter; print(Counter((o["e"], o["v"]) for o in out))
