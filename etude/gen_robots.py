"""Fiche des deux robots pour l'application (app/robots.js) et le dépôt (robots.json) : règles + ce que le rejeu a mesuré.
Rejoue d'abord : bt_swing3.py (TRI=rsi MAXP=5) et bt_day2.py. Variables : VL_SORTIE_JS, VL_SORTIE_JSON, VL_DATE_ETUDE."""
import os, json, numpy as np, pandas as pd, datetime as dt
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
date = os.environ.get("VL_DATE_ETUDE", dt.date.today().isoformat())
sw = json.load(open(f"{D}/robot_swing_rsi_5.json"))
sw2 = pd.read_parquet(f"{D}/bt_swing2.parquet"); sw2 = sw2[sw2.k == "r30_s3.0_n7_s5"]
def stats(g):
    if not len(g): return None
    G = g[g.ret > 0]; P0 = g[g.ret <= 0]
    return dict(n=int(len(g)), win=round(len(G) / len(g) * 100, 1), esp=round(float(g.ret.mean()), 2), mg=round(float(G.ret.mean()), 2) if len(G) else None, mp=round(float(P0.ret.mean()), 2) if len(P0) else None)
an = pd.Timestamp(date).timestamp() - 365 * 86400
swing = dict(nom="Robot swing", regle=dict(rsi_max=30, prix_min=5, liquidite_musd=20, stop_atr=3.0, max_seances=7, max_positions=5, sortie="première clôture au-dessus de la moyenne 5 jours"),
    mesure=dict(P1=dict(sw["tous_signaux"]["P1"], du="2016-11", au="2021-06"), P2=dict(sw["tous_signaux"]["P2"], du="2021-07", au=date[:7]), an=stats(sw2[sw2.t >= an]),
        portefeuille=sw["portefeuille"], temoin=dict(P1=-0.135, P2=-0.077)),
    verdict="avantage" if sw["tous_signaux"]["P1"]["esp"] > 0.2 and sw["tous_signaux"]["P2"]["esp"] > 0.2 else "aucun")
d2 = pd.read_parquet(f"{D}/bt_day2.parquet"); K = "g-10_-5_p5_oNone_s4.0"; g = d2[(d2.k == K)]
def per(x, sl): return stats(x[x.sl == sl])
day = dict(nom="Robot day trading", regle=dict(ecart_min=-10, ecart_max=-5, prix_min=5, liquidite_musd=20, stop_pct=4.0, max_positions=5, sortie="vente 5 minutes avant la clôture"),
    mesure=dict(P1=dict(per(g[g.per == "P1"], 0.0005), du="2024-01", au="2025-05"), P2=dict(per(g[g.per == "P2"], 0.0005), du="2025-06", au=date[:7]),
        P2_ecart_large=per(g[g.per == "P2"], 0.0015), an=per(g[g.t >= an], 0.0005), temoin=dict(P1=-0.094, P2=-0.142)))
p1, p2 = day["mesure"]["P1"]["esp"], day["mesure"]["P2"]["esp"]
day["verdict"] = "avantage" if p1 > 0.2 and p2 > 0.2 else "observation"
out = dict(info=dict(date=date, methode="Règles choisies sur la 1re période, confirmées sur la 2de ; achat au prix vendeur, vente au prix acheteur (écart 0,05 %), frais exclus. Données : actions cotées aujourd'hui (celles qui ont disparu manquent, résultats un peu flatteurs)."), swing=swing, day=day)
js = os.environ.get("VL_SORTIE_JS", f"{D}/robots.js"); jn = os.environ.get("VL_SORTIE_JSON", f"{D}/robots.json")
open(js, "w").write("// généré par gen_robots.py : règles des robots et ce que le rejeu sur l'historique a mesuré\nwindow.VL = window.VL || {}; window.VL.ROBOTS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
json.dump(out, open(jn, "w"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
