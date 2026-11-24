"""Une ligne par action et par seance a partir des barres d'une heure (Yahoo, ~3 ans, pre-marche compris en prix).
Barres pre-marche 4h..9h (la barre de 9h couvre 9h-9h30), seance 9h30, 10h30, ..., 15h30 (heure de New York).
Les ecarts dus aux divisions d'actions (barres horaires non ajustees) sont ecartes grace aux barres journalieres ajustees."""
import os, glob, datetime, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor
NY = ZoneInfo("America/New_York"); D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__)))
CIBLES = [1, 2, 3, 5]

def premier(H, L, p0, x):  # +x avant -x ; meme barre horaire : on ne sait pas -> nan (non tranche)
    up, dn = p0 * (1 + x / 100), p0 * (1 - x / 100)
    for h, l in zip(H, L):
        a, b = h >= up, l <= dn
        if a and b: return np.nan
        if b: return 0.0
        if a: return 1.0
    return -1.0  # ni l'un ni l'autre avant la cloture

def traiter(sym):
    try: A = np.load(f"{D}/data1h/{sym}.npz")["a"]
    except Exception: return []
    dj = {}
    try:
        Dd = np.load(f"{D}/data1d/{sym}.npz")["a"]
        for t, o, h, l, c, v in Dd: dj[datetime.datetime.fromtimestamp(int(t), NY).strftime("%Y-%m-%d")] = (o, c)
    except Exception: pass
    t = A[:, 0].astype(np.int64); dt = [datetime.datetime.fromtimestamp(x, NY) for x in t]
    day = np.array([d.strftime("%Y-%m-%d") for d in dt]); mi = np.array([d.hour * 60 + d.minute for d in dt])
    o, h, l, c, v = A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]
    out = []; prevC = None; prevD = None; hist = []
    cut = np.r_[0, np.where(day[1:] != day[:-1])[0] + 1, len(day)]  # barres deja dans l'ordre du temps
    for a0, a1 in zip(cut[:-1], cut[1:]):
        d = day[a0]; k = np.arange(a0, a1); m = mi[k]
        reg = k[(m >= 570) & (m < 960)]; pre = k[(m >= 240) & (m < 570)]
        if len(reg) < 6 or mi[reg[0]] != 570: prevC = None; continue
        O, C = o[reg[0]], c[reg[-1]]; Hr, Lr = h[reg], l[reg]
        v1 = v[reg[0]]
        if prevC and prevC > 0:
            r = {"sym": sym, "d": d, "prevC": prevC, "O": O, "C": C, "gap": (O / prevC - 1) * 100}
            # garde-fou division d'action : ecart horaire tres different de l'ecart journalier ajuste
            if prevD in dj and d in dj:
                gj = (dj[d][0] / dj[prevD][1] - 1) * 100
                if abs(gj - r["gap"]) > 8: prevC = C; prevD = d; hist.append((v1, C, (Hr.max() / Lr.min() - 1) * 100, v[reg].sum())); continue
            pm = mi[pre]
            def pre_c(mm):  # cloture de la barre de pre-marche commencant a mm
                j = np.where(pm == mm)[0]; return c[pre[j[0]]] if len(j) else np.nan
            r["p800"] = pre_c(420); r["p900"] = pre_c(480); r["p930"] = pre_c(540); r["npre"] = len(pre)
            r["p1030"] = c[reg[0]]; r["H"] = Hr.max(); r["L"] = Lr.min()
            if len(hist) >= 5:
                r["rv1h"] = v1 / max(1, np.mean([x[0] for x in hist[-10:]])); r["dv10"] = np.mean([x[3] for x in hist[-10:]]) * prevC
                r["rng10"] = np.mean([x[2] for x in hist[-10:]])
            r["r1j"] = (prevC / hist[-1][1] - 1) * 100 if len(hist) >= 1 and hist[-1][1] else np.nan
            r["r5j"] = (prevC / hist[-5][1] - 1) * 100 if len(hist) >= 5 else np.nan
            for nom, p0, j0 in (("o", O, 0), ("1030", r["p1030"], 1)):
                if not (p0 > 0): continue
                r[f"rc_{nom}"] = (C / p0 - 1) * 100
                for x in CIBLES: r[f"w{x}_{nom}"] = premier(Hr[j0:], Lr[j0:], p0, x)
            if r["p930"] > 0:
                p0 = r["p930"]; r["ro_930"] = (O / p0 - 1) * 100; r["rc_930"] = (C / p0 - 1) * 100; r["r1030_930"] = (r["p1030"] / p0 - 1) * 100
                for x in CIBLES: r[f"w{x}_930"] = premier(Hr, Lr, p0, x)
            if r["p800"] > 0:
                p0 = r["p800"]; r["r930_800"] = (r["p930"] / p0 - 1) * 100; r["ro_800"] = (O / p0 - 1) * 100; r["rc_800"] = (C / p0 - 1) * 100
                kk = k[(m >= 480) & (m < 960)]
                for x in CIBLES: r[f"w{x}_800"] = premier(h[kk], l[kk], p0, x)
            out.append(r)
        hist.append((v1, C, (Hr.max() / Lr.min() - 1) * 100, v[reg].sum())); prevC = C; prevD = d
    return out

if __name__ == "__main__":
    S = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{D}/data1h/*.npz"))
    os.makedirs(f"{D}/h_parts", exist_ok=True); parts = []
    for i in range(0, len(S), 400):  # par paquets (memoire)
        f = f"{D}/h_parts/p{i:05d}.parquet"; parts.append(f)
        if os.path.exists(f): continue
        R = []
        with ProcessPoolExecutor(2) as ex:
            for L in ex.map(traiter, S[i:i + 400], chunksize=10): R.extend(L)
        x = pd.DataFrame(R); fl = x.select_dtypes("float64").columns; x[fl] = x[fl].astype("float32"); x.to_parquet(f); print(i + 400, flush=True)
    df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    df.to_parquet(f"{D}/jours1h.parquet"); print(len(S), "actions,", len(df), "seances", df.d.nunique(), "jours", df.d.min(), df.d.max())
