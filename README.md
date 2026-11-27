# Veille Live : données

Ce dépôt sert uniquement à mettre à jour le tableau des **scénarios mesurés** et l'étude des **deux robots** de l'application Veille Live.

- `scenarios.json` : le tableau lu par l'application (une fois par semaine au plus, en lecture seule).
  Chaque ligne décrit une situation (écart d'ouverture, pré-marché, veille, Nasdaq, jour de résultats…) et ce qui s'est passé
  toutes les fois où elle s'est produite : part des séances finies au-dessus du prix d'achat, probabilité réaliste, résultat moyen.
- `robots.json` : les règles du robot swing et du robot day trading, et ce que leur rejeu sur l'historique a mesuré (deux périodes, 12 derniers mois, témoin au hasard), refait chaque semaine.
- `etude/` : les scripts qui refont l'étude (données gratuites Yahoo Finance et calendrier des résultats Nasdaq, environ 5 800 actions
  américaines, 10 ans en journalier et 2 ans en barres d'une heure). `bash etude/maj_hebdo.sh` refait tout (environ 1 h 30).

Aucune clé, aucun mot de passe, aucune donnée personnelle ici. Analyse automatique du passé, pas un conseil.
