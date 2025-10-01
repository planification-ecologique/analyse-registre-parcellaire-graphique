# Analyse du Registre Parcellaire Graphique (RPG)

Ensemble de fonctions utilitaires Python et de notebooks Jupyter pour analyser l'évolution des rotations de cultures et des couverts en France à partir du RPG (Registre Parcellaire Graphique).

## Objectifs
- Extraire et comparer des millésimes du RPG (ex. 2023 vs 2024)
- Étudier l'évolution des occupations du sol, rotations de cultures et couverts
- Produire des tableaux de synthèse et visualisations spatiales

## Contenu du dépôt
- `data/` : jeux de données RPG en format GeoPackage (`.gpkg`)
  - `PARCELLES_GRAPHIQUES_2023.gpkg`
  - `PARCELLES_GRAPHIQUES_2024.gpkg`
- `notebook/` : notebooks d'analyse
  - `RPG.ipynb` : notebook principal d'exploration et de comparaison multi-années

## Prérequis
- Python ≥ 3.10
- Outils et bibliothèques conseillés:
  - `geopandas`, `pandas`, `pyogrio` ou `fiona`, `shapely`
  - `jupyter`/`jupyterlab`
  - `matplotlib`/`seaborn`

## Installation rapide
```bash
# Créer et activer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # sous macOS/Linux

# Installer les dépendances minimales
pip install --upgrade pip
pip install geopandas pandas pyogrio shapely jupyter matplotlib seaborn
```

Astuce: sur macOS, il peut être utile d'installer `gdal` via Homebrew avant `geopandas/pyogrio`:
```bash
brew install gdal
```

## Utilisation
1. Placer les fichiers RPG `.gpkg` dans le dossier `data/` (déjà présent pour 2023 et 2024).
2. Lancer Jupyter et ouvrir le notebook principal:
```bash
jupyter lab notebook/RPG.ipynb
```
3. Exécuter les cellules pour:
   - Charger les couches `PARCELLES_GRAPHIQUES_20XX`
   - Harmoniser les schémas attributaires (codes cultures, etc.)
   - Comparer les millésimes et calculer des indicateurs de rotation
   - Cartographier les changements

## Structure (simplifiée)
```
.
├── data/
│   ├── PARCELLES_GRAPHIQUES_2023.gpkg
│   └── PARCELLES_GRAPHIQUES_2024.gpkg
└── notebook/
    └── RPG.ipynb
```

## Notes sur les données
- Les fichiers `.gpkg` proviennent du RPG (source officielle). Vérifiez les métadonnées et licences d'utilisation associées aux millésimes.
- Les champs d'intérêt communs: identifiants de parcelles, codes cultures, surfaces, géométries.

## Licence
Sauf mention contraire, le code de ce dépôt est publié sous licence MIT. Les données RPG restent soumises à leurs licences et conditions d'utilisation propres.

## Citation
Si vous utilisez ce dépôt dans un travail ou une publication, merci de citer: « Analyse des rotations agricoles à partir du RPG (France), notebook et utilitaires Python » et référencer la source des données RPG.

## Contact
Pour toute question ou suggestion, ouvrez une issue sur le dépôt ou proposez une *pull request*.

## PostGIS avec Docker Compose (optionnel)

Pour accélérer les analyses spatiales, vous pouvez charger les GeoPackages RPG dans une base PostGIS dockerisée.

### Démarrage
```bash
docker compose up -d postgis
```

### Chargement automatique des GPKG
Placez vos fichiers `PARCELLES_GRAPHIQUES_*.gpkg` dans `data/` (déjà présent pour 2023 et 2024), puis lancez le service de chargement:
```bash
docker compose up loader
```
Le script `docker/scripts/load_rpg.sh` importe chaque couche avec GDAL/OGR (`ogr2ogr`) vers le schéma `rpg` de la base `rpg`. Les tables sont suffixées par l'année détectée (ex: `parcelles_graphiques_2023`).

### Connexion
```bash
PGPASSWORD=rpg psql -h localhost -p 5432 -U rpg -d rpg
```

Paramètres par défaut:
- Base: `rpg`
- Utilisateur: `rpg`
- Mot de passe: `rpg`
- Schéma: `rpg`

Extensions et schéma sont initialisés via `docker/initdb/01_init.sql`.
