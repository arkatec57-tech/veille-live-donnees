"""Historique des publications de resultats (calendrier Nasdaq, un jour par requete) : symbole, date, BPA publie, BPA attendu, surprise.
Sortie : resultats.parquet. L'heure (avant l'ouverture / apres la cloture) n'est pas fournie : le jour de reaction se deduit des cours."""
import json, os, time, random, urllib.request, datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
D = os.environ.get("VL_ETUDE", os.path.dirname(os.path.abspath(__file__))); os.makedirs(f"{D}/resu", exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Accept": "application/json"}
def un(d):
    f = f"{D}/resu/{d}.json"
    if os.path.exists(f): return
    for k in range(4):
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://api.nasdaq.com/api/calendar/earnings?date={d}", headers=UA), timeout=40).read())
            rows = (j.get("data") or {}).get("rows") or []
            json.dump(rows, open(f, "w")); return
        except Exception:
            time.sleep(3 * (k + 1) + random.random())
def num(x):
    try: return float(str(x).replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
    except Exception: return None
if __name__ == "__main__":
    d0, d1 = datetime.date(2016, 1, 1), datetime.date.today()
    jours = [(d0 + datetime.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1) if (d0 + datetime.timedelta(i)).weekday() < 5]
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(un, jours)):
            if i % 200 == 0: print(i, len(jours), round(time.time() - t0), "s", flush=True)
    L = []
    for d in jours:
        f = f"{D}/resu/{d}.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            L.append({"sym": r.get("symbol"), "d": d, "eps": num(r.get("eps")), "prev": num(r.get("epsForecast")), "surp": num(r.get("surprise")), "nest": num(r.get("noOfEsts"))})
    df = pd.DataFrame(L); df.to_parquet(f"{D}/resultats.parquet"); print(len(df), "publications,", df.sym.nunique(), "actions,", df.d.min(), "->", df.d.max())
