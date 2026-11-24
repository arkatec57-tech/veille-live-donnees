"""Une ligne par action et par seance, a partir des barres de 5 minutes (heure de New York), sans regarder le futur pour les
caracteristiques : tout ce qui sert a decrire la situation est connu au moment de l'entree. Les resultats (apres l'entree)
sont dans des colonnes separees. Sortie : jours5m.parquet
"""
import os, sys, json, glob, datetime
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor
NY = ZoneInfo("America/New_York"); D = os.path.dirname(os.path.abspath(__file__))
CIBLES = [1, 2, 3, 5, 10]  # % (sur l'action ; un ETF x3 fait environ 3 fois plus)

def premier(H, L, p0, x, fin=None):
    """+x % avant -x % a partir du prix p0, sur les barres (H, L) ; meme barre : compte comme perdant (prudence).
    Renvoie 1 (objectif), 0 (stop), nan (ni l'un ni l'autre)."""
    up, dn = p0 * (1 + x / 100), p0 * (1 - x / 100)
    for h, l in zip(H, L):
        if l <= dn: return 0.0
        if h >= up: return 1.0
    return np.nan

def mfe(H, L, p0):
    return (np.max(H) / p0 - 1) * 100 if len(H) else np.nan, (np.min(L) / p0 - 1) * 100 if len(L) else np.nan

def traiter(path):
    sym = os.path.basename(path)[:-4]
    try: A = np.load(path)["a"]
    except Exception: return []
    if len(A) < 200: return []
    t = A[:, 0].astype(np.int64)
    dt = [datetime.datetime.fromtimestamp(x, NY) for x in t]
    day = np.array([d.strftime("%Y-%m-%d") for d in dt]); mi = np.array([d.hour * 60 + d.minute for d in dt])
    o, h, l, c, v = A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]
    jours = sorted(set(day)); out = []
    prevC = None; hist = []  # historique des seances precedentes (volume 15 premieres min, amplitude)
    for d in jours:
        k = np.where(day == d)[0]
        m = mi[k]; reg = k[(m >= 570) & (m < 960)]; pre = k[(m >= 240) & (m < 570)]
        if len(reg) < 60: prevC = None; continue
        r = {"sym": sym, "d": d}
        O = o[reg[0]]; C = c[reg[-1]]
        # seance : barres a partir de l'ouverture
        Hr, Lr, Cr, Vr, Mr = h[reg], l[reg], c[reg], v[reg], mi[reg]
        def px_a(mm):  # cours de cloture de la derniere barre terminee avant la minute mm (seance)
            j = np.where(Mr + 5 <= mm)[0]; return Cr[j[-1]] if len(j) else np.nan
        vol15 = Vr[Mr < 585].sum(); vol5 = Vr[Mr < 575].sum(); vjour = Vr.sum()
        rng = (Hr.max() / Lr.min() - 1) * 100
        if prevC and prevC > 0:
            r["prevC"] = prevC; r["O"] = O; r["C"] = C
            r["gap"] = (O / prevC - 1) * 100
            # pre-marche (prix seulement : Yahoo ne donne pas les volumes du pre-marche)
            if len(pre):
                pm, pc_, ph, pl = mi[pre], c[pre], h[pre], l[pre]
                def pre_a(mm):
                    j = np.where(pm + 5 <= mm)[0]; return pc_[j[-1]] if len(j) else np.nan
                r["p800"] = pre_a(480); r["p900"] = pre_a(540); r["p925"] = pre_a(565); r["p830"] = pre_a(510)
                r["npre"] = len(pre)  # nombre de barres de 5 min avec echanges (activite du pre-marche)
                r["npre_830"] = int(((pm >= 510) & (pm < 565)).sum())
                r["preH"] = ph[pm < 565].max() if (pm < 565).any() else np.nan; r["preL"] = pl[pm < 565].min() if (pm < 565).any() else np.nan
            else:
                r["npre"] = 0; r["npre_830"] = 0
            # volume relatif de l'ouverture (15 premieres minutes) par rapport aux 10 seances precedentes
            if len(hist) >= 5:
                r["rv15"] = vol15 / max(1, np.mean([x[0] for x in hist[-10:]]))
                r["rv5"] = vol5 / max(1, np.mean([x[1] for x in hist[-10:]]))
                r["rng10"] = np.mean([x[2] for x in hist[-10:]])  # amplitude habituelle d'une seance (%)
                r["vj10"] = np.mean([x[3] for x in hist[-10:]]); r["dv10"] = r["vj10"] * prevC
            r["r1j"] = hist[-1][4] if hist else np.nan  # variation de la veille (%)
            r["r5j"] = (prevC / hist[-5][5] - 1) * 100 if len(hist) >= 5 else np.nan
            r["p935"] = px_a(575); r["p945"] = px_a(585); r["p1000"] = px_a(600); r["p1030"] = px_a(630); r["p1100"] = px_a(660)
            if not (Mr < 585).any() or Mr[0] > 575: hist.append((vol15, vol5, rng, vjour, (C / prevC - 1) * 100, C)); prevC = C; continue  # ouverture manquante
            r["or15H"] = Hr[Mr < 585].max(); r["or15L"] = Lr[Mr < 585].min()
            r["C"] = C; r["H"] = Hr.max(); r["L"] = Lr.min()
            # resultats a partir de plusieurs entrees : ouverture, 9h45, cours de pre-marche a 9h25 (jusqu'a l'ouverture puis la seance)
            for nom, p0, deb in (("o", O, 570), ("945", r["p945"], 585), ("1000", r["p1000"], 600)):
                if not (p0 > 0): continue
                j = Mr >= deb
                for x in CIBLES: r[f"w{x}_{nom}"] = premier(Hr[j], Lr[j], p0, x)
                r[f"rc_{nom}"] = (C / p0 - 1) * 100
                for fin, lib in ((630, "1h"), (600, "30")):
                    if fin > deb: r[f"r{lib}_{nom}"] = (px_a(fin + (0 if deb == 570 else deb - 570)) / p0 - 1) * 100 if True else np.nan
                r[f"mfe_{nom}"], r[f"mae_{nom}"] = mfe(Hr[j], Lr[j], p0)
            # entree a 9h25 en pre-marche : on inclut la fin du pre-marche puis toute la seance
            if r.get("p925", np.nan) > 0:
                p0 = r["p925"]; kk = k[(m >= 565) & (m < 960)]
                for x in CIBLES: r[f"w{x}_925"] = premier(h[kk], l[kk], p0, x)
                r["rc_925"] = (C / p0 - 1) * 100; r["ro_925"] = (O / p0 - 1) * 100; r["r1h_925"] = (px_a(630) / p0 - 1) * 100
            # entree a 8h00 en pre-marche (gros mouvement tot) : resultat a 9h25, a l'ouverture, a 10h, a la cloture
            if r.get("p800", np.nan) > 0:
                p0 = r["p800"]; r["r925_800"] = (r.get("p925", np.nan) / p0 - 1) * 100; r["ro_800"] = (O / p0 - 1) * 100
                r["r1000_800"] = (r["p1000"] / p0 - 1) * 100; r["rc_800"] = (C / p0 - 1) * 100
                kk = k[(m >= 480) & (m < 960)]
                for x in CIBLES: r[f"w{x}_800"] = premier(h[kk], l[kk], p0, x)
            # demarrage en seance (« pepite ») : premiere barre apres 9h45 et avant 15h ou l'action depasse +S % sur la veille
            # avec une barre de volume au moins 3 fois la moyenne des barres precedentes de la seance ; entree a l'ouverture de la barre suivante
            for S in (5, 10, 20):
                idx = None
                for ii in range(3, len(reg) - 1):
                    if Mr[ii] < 585 or Mr[ii] >= 900: continue
                    if Cr[ii] / prevC - 1 >= S / 100 and Vr[ii] >= 3 * max(1, Vr[:ii].mean()):
                        idx = ii; break
                if idx is None: continue
                p0 = o[reg[idx + 1]]; j = slice(idx + 1, len(reg))
                r[f"dem{S}_m"] = int(Mr[idx]); r[f"dem{S}_x"] = Vr[idx] / max(1, Vr[:idx].mean()); r[f"dem{S}_pc"] = (Cr[idx] / prevC - 1) * 100
                r[f"dem{S}_ro"] = (Cr[idx] / O - 1) * 100  # deja fait depuis l'ouverture
                for x in CIBLES: r[f"w{x}_dem{S}"] = premier(Hr[j], Lr[j], p0, x)
                r[f"rc_dem{S}"] = (C / p0 - 1) * 100
                e30 = [q for q in range(idx + 1, len(reg)) if Mr[q] >= Mr[idx + 1] + 30]
                e60 = [q for q in range(idx + 1, len(reg)) if Mr[q] >= Mr[idx + 1] + 60]
                r[f"r30_dem{S}"] = (Cr[e30[0] - 1] / p0 - 1) * 100 if e30 else np.nan
                r[f"r60_dem{S}"] = (Cr[e60[0] - 1] / p0 - 1) * 100 if e60 else np.nan
            out.append(r)
        hist.append((vol15, vol5, rng, vjour, (C / prevC - 1) * 100 if prevC else np.nan, C))
        prevC = C
    return out

if __name__ == "__main__":
    F = sorted(glob.glob(f"{D}/data5m/*.npz"))
    R = []
    with ProcessPoolExecutor(2) as ex:
        for i, L in enumerate(ex.map(traiter, F, chunksize=20)):
            R.extend(L)
    df = pd.DataFrame(R); df.to_parquet(f"{D}/jours5m.parquet"); print(len(F), "actions,", len(df), "seances", df["d"].nunique(), "jours")
