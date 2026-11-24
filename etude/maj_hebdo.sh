#!/bin/bash
# Mise a jour hebdomadaire du tableau des scenarios de Veille Live (tache programmee) :
# collecte (liste Nasdaq, Yahoo journalier 10 ans et horaire 2 ans, calendrier des resultats), rejeu, synthese, scenarios.json.
# Usage : bash maj_hebdo.sh <dossier de travail>   (environ 1 h 30)
set -e
ICI="$(cd "$(dirname "$0")" && pwd)"; W="${1:-$ICI/travail}"; mkdir -p "$W"
cp "$ICI"/*.py "$W"/; cd "$W"; export VL_ETUDE="$W"
pip install -q --break-system-packages pyarrow pandas numpy >/dev/null 2>&1 || true
python3 fetch.py univers
python3 fetch.py 1d 3 > fetch1d.log 2>&1
python3 fetch.py 1h 3 > fetch1h.log 2>&1
python3 resultats.py > resultats.log 2>&1
python3 quot.py && python3 quot_scen.py > quot_out.txt
python3 heures.py > heures.log && python3 scen_h.py > scen_h.log && python3 analyse.py scenarios_h.csv synthese_h.json > analyse_h.log
python3 resu_scen.py > resu_scen.log
python3 export.py > export.log && python3 gen_app.py
python3 - <<'PY'
import json; d = json.load(open("scenarios.json")); n = len(d["L"])
assert n > 100, f"tableau trop court ({n})"; print("tableau", d["info"], n, "scenarios")
PY
