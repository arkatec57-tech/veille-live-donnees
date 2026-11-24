"""Collecte : univers (liste Nasdaq), barres 5 min sur 60 jours (pre-marche compris) et barres journalieres 10 ans (Yahoo).
Usage : python3 fetch.py univers | python3 fetch.py 5m | python3 fetch.py 1d
"""
import json, sys, os, time, urllib.request, threading, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Accept": "application/json"}
D = os.path.dirname(os.path.abspath(__file__))

def get(u, essais=4):
    for k in range(essais):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40).read())
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(3 * (k + 1) + random.random() * 2)
        except Exception:
            time.sleep(2 * (k + 1))
    return None

def univers():
    s = get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0&download=true")
    rows = s["data"]["rows"]; L = []
    for r in rows:
        sym = r["symbol"].strip().replace("/", "-")
        if not sym or "^" in sym or " " in sym: continue
        try: px = float(r["lastsale"].replace("$", "").replace(",", "")); vol = float(r["volume"] or 0); mc = float(r["marketCap"] or 0)
        except Exception: continue
        if px < 1 or vol <= 0: continue
        L.append((sym, px, vol, px * vol, mc))
    L.sort(key=lambda x: -x[3])
    sel = [x[0] for x in L[:2000]]
    for x in ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "META", "AMZN", "GOOGL", "PLTR", "MSTR", "COIN", "SMCI", "AVGO", "NFLX", "MU", "INTC", "SPY", "QQQ", "IWM"]:
        if x not in sel: sel.append(x)
    json.dump({"syms": sel, "info": {x[0]: [x[1], x[2], x[4]] for x in L}}, open(f"{D}/univers.json", "w"))
    print(len(rows), "lignes Nasdaq,", len(L), "a 1 $ et plus,", len(sel), "retenues")

def un(sym, mode):
    out = f"{D}/data{mode}/{sym}.npz"
    if os.path.exists(out): return "deja"
    u = {"5m": f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=60d&interval=5m&includePrePost=true",
         "1h": f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=730d&interval=60m&includePrePost=true",
         "1d": f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=10y&interval=1d"}[mode]
    j = get(u)
    if mode == "1h" and not (j and j.get("chart", {}).get("result")):  # introduction en bourse recente : historique plus court
        for rg in ("1y", "6mo"):
            j = get(u.replace("range=730d", "range=" + rg), essais=2)
            if j and j.get("chart", {}).get("result"): break
    try: y = j["chart"]["result"][0]; q = y["indicators"]["quote"][0]; t = y["timestamp"]
    except Exception: return "vide"
    A = np.array([[t[i], q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i] or 0] for i in range(len(t))
                  if q["close"][i] is not None and q["high"][i] is not None and q["low"][i] is not None and q["open"][i] is not None], dtype=np.float64)
    if mode == "1d":  # cloture ajustee des dividendes et divisions : on rapporte o/h/l au meme facteur
        try:
            adj = y["indicators"]["adjclose"][0]["adjclose"]; ac = np.array([adj[i] for i in range(len(t)) if q["close"][i] is not None and q["high"][i] is not None and q["low"][i] is not None and q["open"][i] is not None], dtype=np.float64)
            f = ac / A[:, 4]; A[:, 1:5] *= f[:, None]
        except Exception: pass
    if not len(A): return "vide"
    np.savez_compressed(out, a=A.astype(np.float64))
    return len(A)

def lot(mode, n_thr=3):
    os.makedirs(f"{D}/data{mode}", exist_ok=True)
    syms = json.load(open(f"{D}/univers.json"))["syms"]
    t0 = time.time(); fait = [0]; lk = threading.Lock(); vides = []
    def w(s):
        r = un(s, mode)
        with lk:
            fait[0] += 1
            if r == "vide": vides.append(s)
            if fait[0] % 100 == 0: print(mode, fait[0], "/", len(syms), round(time.time() - t0), "s", "vides", len(vides), flush=True)
    with ThreadPoolExecutor(n_thr) as ex: list(ex.map(w, syms))
    print(mode, "fini", fait[0], "vides", len(vides), vides[:30], round(time.time() - t0), "s", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "univers": univers()
    else: lot(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
