"""Synthese : pour chaque scenario, chiffres sur les deux periodes, verdict de stabilite, et (ecarts d'ouverture) controle sur 10 ans.
Sortie : synthese.json (pour la page) et resume texte."""
import os
import pandas as pd, numpy as np, json, re
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "scenarios.csv"; DST = sys.argv[2] if len(sys.argv) > 2 else "synthese.json"
R = pd.read_csv(f"{D}/{SRC}")
Q = pd.read_csv(f"{D}/quot_scen.csv") if __import__("os").path.exists(f"{D}/quot_scen.csv") else None
# probabilite « objectif avant stop » parmi les cas tranches (le reste : ni l'un ni l'autre avant la cloture)
R["dec"] = R.p_hit + R.p_stop
R["p_dir"] = R.p_hit / R.dec.replace(0, np.nan)
# erreur-type approchee de p_dir (delta) : se_hit / dec
R["se_dir"] = R.se_hit / R.dec.replace(0, np.nan)
out = []
for (fam, ent, lib), g in R.groupby(["fam", "ent", "lib"], sort=False):
    A = g[g.per == "A"]; T = g[g.per == "T"]
    if not len(A) or not len(T): continue
    A, T = A.iloc[0], T.iloc[0]
    n = A.n + T.n
    if n < 40: continue
    # chiffres combines (moyenne ponderee) et erreurs-types combinees
    w = lambda k: (A[k] * A.n + T[k] * T.n) / n
    se = lambda k: np.sqrt((A[k] * A.n) ** 2 + (T[k] * T.n) ** 2) / n
    p_up, s_up = w("p_up"), se("se_up"); moy, s_moy = w("moy"), se("se_moy")
    dec = w("dec"); p_dir = (A.p_hit * A.n + T.p_hit * T.n) / max(1e-9, (A.dec * A.n + T.dec * T.n)); s_dir = se("se_hit") / max(dec, 1e-9)
    # reference : toutes les actions a la meme heure d'entree (meme periode), sinon pile ou face
    ref_up = w("base_up") if not np.isnan(A.get("base_up", np.nan)) else 0.5
    zA = (A.p_up - (A.base_up if not np.isnan(A.base_up) else 0.5)) / A.se_up if A.se_up > 0 else 0
    zT = (T.p_up - (T.base_up if not np.isnan(T.base_up) else 0.5)) / T.se_up if T.se_up > 0 else 0
    zdA = (A.p_dir - 0.5) / A.se_dir if A.se_dir > 0 else 0; zdT = (T.p_dir - 0.5) / T.se_dir if T.se_dir > 0 else 0
    zmA = A.moy / A.se_moy if A.se_moy > 0 else 0; zmT = T.moy / T.se_moy if T.se_moy > 0 else 0
    def verdict(za, zt):
        if np.sign(za) == np.sign(zt) and abs(za) >= 2 and abs(zt) >= 2: return "confirme"
        if np.sign(za) == np.sign(zt) and abs(za) >= 1 and abs(zt) >= 1: return "indice"
        return "non"
    vUp = verdict(zA, zT); vDir = verdict(zdA, zdT); vMoy = verdict(zmA, zmT)
    sens = None
    for v, z in ((vUp, zA + zT), (vDir, zdA + zdT), (vMoy, zmA + zmT)):
        if v != "non": sens = "haut" if z > 0 else "bas"; break
    force = "confirme" if "confirme" in (vUp, vDir, vMoy) else "indice" if "indice" in (vUp, vDir, vMoy) else "non"
    # stabilite : aucune mesure ne doit aller nettement dans l'autre sens sur l'une des deux periodes
    if sens:
        s0 = 1 if sens == "haut" else -1
        if any(z * s0 <= -1 for z in (zA, zT, zdA, zdT, zmA, zmT)): force = "instable"
    o = dict(fam=fam, ent=ent, lib=lib, cible=int(A.cible), n=int(n), jours=int(A.jours + T.jours), actions=int(max(A.actions, T.actions)),
             p_up=p_up, ic_up=1.645 * s_up, ref_up=ref_up, p_dir=p_dir, ic_dir=1.645 * s_dir, dec=dec, moy=moy, ic_moy=1.645 * s_moy,
             gain=w("gain_moy"), perte=w("perte_moy"), med=w("med"),
             A=dict(n=int(A.n), p_up=A.p_up, p_dir=A.p_dir, moy=A.moy), T=dict(n=int(T.n), p_up=T.p_up, p_dir=T.p_dir, moy=T.moy),
             z=dict(up=[zA, zT], dir=[zdA, zdT], moy=[zmA, zmT]), force=force, sens=sens)
    # controle sur 10 ans (ecart d'ouverture, achat a l'ouverture, resultat a la cloture)
    if Q is not None and ent == "o" and not fam.startswith("F"):
        m = re.match(r"écart (?:d'ouverture )?(.+?)(?: · (.*))?$", lib)
        if m:
            cond = m.group(2) or ""
            key = "écart " + m.group(1) + ({"action à moins de 5 $": " · moins de 5 $", "action à 20 $ et plus": " · 20 $ et plus", "veille en forte hausse (≥ +5 %)": " · veille ≥ +5 %",
                                            "veille en forte baisse (≤ −5 %)": " · veille ≤ −5 %", "Nasdaq (QQQ) en hausse à l'ouverture": " · Nasdaq en hausse", "Nasdaq (QQQ) en baisse à l'ouverture": " · Nasdaq en baisse", "": ""}.get(cond, "§"))
            q = Q[Q.lib == key]
            if len(q):
                qa, qt, qs = (q[q.per == p] for p in ("A", "T", "*"))
                if len(qs):
                    o["dix"] = dict(n=int(qs.n.iloc[0]), p_up=float(qs.p_up.iloc[0]), ic=1.645 * float(qs.se_up.iloc[0]), moy=float(qs.moy.iloc[0]),
                                    pA=float(qa.p_up.iloc[0]) if len(qa) else None, pT=float(qt.p_up.iloc[0]) if len(qt) else None,
                                    hit3=float(qs.hit3.iloc[0]), low3=float(qs.low3.iloc[0]))
                    zq = [(float(x.p_up.iloc[0]) - 0.5) / float(x.se_up.iloc[0]) if len(x) else 0 for x in (qa, qt)]
                    o["dix"]["force"] = verdict(*zq)
    out.append(o)
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)): return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)): return int(x)
    return x
out = clean(out)
meta = json.load(open(f"{D}/meta.json")) if __import__("os").path.exists(f"{D}/meta.json") else {}
json.dump({"meta": meta, "S": out}, open(f"{D}/{DST}", "w"), ensure_ascii=False)
print(len(out), "scenarios ;", sum(o["force"] == "confirme" for o in out), "confirmes ;", sum(o["force"] == "indice" for o in out), "indices ;", sum(o["force"] == "instable" for o in out), "instables")
for o in out:
    if o["force"] in ("confirme",):
        print(f'{o["fam"][:12]:12} {o["lib"][:75]:75} n={o["n"]:6} up={o["p_up"]*100:4.0f}% (ref {o["ref_up"]*100:3.0f}) dir={o["p_dir"]*100 if o["p_dir"] else 0:4.0f}% moy={o["moy"]:+.2f} A={o["A"]["p_up"]*100:3.0f}/{o["T"]["p_up"]*100:3.0f} {o["sens"]}',
              ("| 10 ans " + f'{o["dix"]["p_up"]*100:.0f}% ({o["dix"]["force"]})') if o.get("dix") else "")
# etalonnage : la probabilite mesuree sur la 1re periode annonce-t-elle bien ce qui s'est passe sur la 2e ?
cal = []
for o in out:
    if o["A"]["n"] >= 100 and o["T"]["n"] >= 50: cal.append((o["A"]["p_up"], o["T"]["p_up"], o["T"]["n"], o["ref_up"]))
C = pd.DataFrame(cal, columns=["pa", "pt", "nt", "ref"])
C["b"] = pd.cut(C.pa, [0, .3, .4, .45, .5, .55, .6, .7, 1])
E = C.groupby("b", observed=True).apply(lambda g: pd.Series({"annonce": g.pa.mean(), "constate": np.average(g.pt, weights=g.nt), "scenarios": len(g)}))
print(E.round(3))
# avec un retrait vers la moyenne (on ne garde qu'une partie de l'ecart mesure)
for k in (1, 0.5, 0.3):
    pr = C.ref + k * (C.pa - C.ref); print("retrait", k, "erreur moyenne", round(float(np.average(np.abs(pr - C.pt), weights=C.nt)) * 100, 2), "points")
json.dump({"cal": [{"annonce": float(r.annonce), "constate": float(r.constate), "n": int(r.scenarios)} for r in E.itertuples()]}, open(f"{D}/etal_{DST}", "w"))
# ecart au marche (on retire la tendance generale de chaque periode, imprevisible)
R0 = pd.read_csv(f"{D}/{SRC}"); X = []
for (fam, ent, lib), g in R0.groupby(["fam", "ent", "lib"], sort=False):
    a, t = g[g.per == "A"], g[g.per == "T"]
    if len(a) and len(t) and a.n.iloc[0] >= 100 and t.n.iloc[0] >= 50 and not np.isnan(a.base_up.iloc[0]):
        X.append((a.p_up.iloc[0] - a.base_up.iloc[0], t.p_up.iloc[0] - t.base_up.iloc[0], t.n.iloc[0]))
X = pd.DataFrame(X, columns=["ea", "et", "nt"]); X["b"] = pd.cut(X.ea, [-1, -.1, -.05, -.02, .02, .05, .1, 1])
print("ecart au marche :", X.groupby("b", observed=True).apply(lambda g: f"annonce {g.ea.mean()*100:+.1f} constate {np.average(g.et, weights=g.nt)*100:+.1f} ({len(g)})").to_dict())
print("correlation", round(float(np.corrcoef(X.ea, X.et)[0, 1]), 2))
