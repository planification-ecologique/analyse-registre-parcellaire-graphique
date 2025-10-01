#%% Importation des bibliotheques
import pandas as pd
from collections import defaultdict
#%% Chargement des fichiers

# Charger les deux fichiers CSV
#df0 = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\RPG csv avec codegrp\rpg 2021.csv")
df1 = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\RPG csv avec codegrp\rpg avec geo 2024.csv")
df2 = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\RPG csv avec codegrp\rpg avec geo 2023.csv")


# Standardiser les noms de colonnes
df1.columns = df1.columns.str.strip().str.upper()
df2.columns = df2.columns.str.strip().str.upper()

#glossaire des codes culture avec leur classification type parcel
liste_codes_cultu = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\Glossaire et classification codes cultu.csv",sep=";", encoding="utf-8")

# Standardiser les colonnes du glossaire
liste_codes_cultu.columns = liste_codes_cultu.columns.str.strip().str.upper()

#%% tests sur la coherence des donnees
# Surface totale en année 1 pour la culture BTH
surf_BTH_1 = df1[df1["CODE_CULTU"] == "BTH"]["SURF_PARC"].sum()
print(f"Surface totale en année 1 cultivée en BTH : {surf_BTH_1} ha")

# Surface totale en année 2 pour la culture BTH
surf_BTH_2 = df2[df2["CODE_CULTU"] == "BTH"]["SURF_PARC"].sum()
print(f"Surface totale en année 2 cultivée en BTH : {surf_BTH_2} ha")

# Surface totale en 2022 pour la culture MIS
surf_MIS_1 = df1[(df1["CODE_CULTU"] == "MIS")|(df1["CODE_CULTU"] == "MIE")]["SURF_PARC"].sum()
print(f"Surface totale en année 1 cultivée en MIS ou MIE : {surf_MIS_1} ha")

# Surface totale en 2023 pour la culture MIS
surf_MIS_2 = df2[(df2["CODE_CULTU"] == "MIS")|(df2["CODE_CULTU"] == "MIE")]["SURF_PARC"].sum()
print(f"Surface totale en année 2 cultivée en MIS ou MIE : {surf_MIS_2} ha")

print(len(df1))
print(len(df2))

#%% test1 sur ID PARCEL
# Vérifier les identifiants de parcelles communs
parcelles_communes = set(df1["ID_PARCEL"]).intersection(df2["ID_PARCEL"])

print(f"Nombre de parcelles communes : {len(parcelles_communes)}")

# Tirer un échantillon aléatoire de 100 parcelles (ou moins si pas assez)
echantillon = pd.Series(list(parcelles_communes)).sample(n=min(100, len(parcelles_communes)), random_state=42)

# Extraire les infos pour ces parcelles dans chaque dataframe
df1_sample = df1[df1["ID_PARCEL"].isin(echantillon)][["ID_PARCEL", "SURF_PARC"]].rename(columns={"SURF_PARC": "SURF_PARC_2022"})
df2_sample = df2[df2["ID_PARCEL"].isin(echantillon)][["ID_PARCEL", "SURF_PARC"]].rename(columns={"SURF_PARC": "SURF_PARC_2023"})

# Fusionner pour comparer
comparaison = df1_sample.merge(df2_sample, on="ID_PARCEL", how="inner")

# Afficher les premières lignes
print(comparaison.head(20))


#%%  Filtrage cultures annuelles 
# Ajouter le type de culture à df1 et df2
df1 = df1.merge(
    liste_codes_cultu[["CODE_CULTURE", "TYPE PARCEL"]],
    left_on="CODE_CULTU",
    right_on="CODE_CULTURE",
    how="left"
).rename(columns={"TYPE PARCEL": "TYPE_PARCEL_1"}).drop(columns=["CODE_CULTURE"])

df2 = df2.merge(
    liste_codes_cultu[["CODE_CULTURE", "TYPE PARCEL"]],
    left_on="CODE_CULTU",
    right_on="CODE_CULTURE",
    how="left"
).rename(columns={"TYPE PARCEL": "TYPE_PARCEL_2"}).drop(columns=["CODE_CULTURE"])

# Définir les types de culture à exclure (non annuelles)
cultures_non_annuelles = [
    "PRAIRIE TEMPORAIRE","JACHERE","PRAIRIE",
    "CULTURE PERENNE","ARBORICULTURE","AROMA","AUTRE"
]

# Filtrer uniquement les cultures annuelles
df1_annuelles = df1[~df1["TYPE_PARCEL_1"].isin(cultures_non_annuelles)]
df2_annuelles = df2[~df2["TYPE_PARCEL_2"].isin(cultures_non_annuelles)]
#%% Verif surface totale des cultures annuelles
import os

# Nom/fichier de sortie
output_path = r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\Transitions\Verif cultures annuelles 2024.xlsx"

# Vérifier que df1_annuelles existe
if 'df1_annuelles' not in globals():
    raise RuntimeError("df1_annuelles n'existe pas — place ce bloc après la création de df1_annuelles.")

# S'assurer que la colonne SURF_PARC est numérique (coerce errors -> NaN puis 0)
df1_annuelles["SURF_PARC"] = pd.to_numeric(df1_annuelles["SURF_PARC"], errors="coerce").fillna(0)

# Surface totale des parcelles conservées comme 'annuelles' (somme simple)
surface_totale_annuelles = df1_annuelles["SURF_PARC"].sum()
print(f"Surface totale des cultures annuelles (après filtrage) : {surface_totale_annuelles}")

# Agrégation : surface cumulée par CODE_CULTU (ne garder que les codes non nuls)
agg_by_culture = (
    df1_annuelles
    .dropna(subset=["CODE_CULTU"])                # on ne veut que les codes valides dans la liste
    .groupby("CODE_CULTU", as_index=False)["SURF_PARC"]
    .sum()
    .rename(columns={"SURF_PARC": "SURFACE_TOTALE"})
    .sort_values(by="SURFACE_TOTALE", ascending=False)
)

# Garder seulement les deux colonnes demandées pour l'export final, si tu veux strictement deux colonnes :
export_table = agg_by_culture[["CODE_CULTU", "SURFACE_TOTALE"]].copy()

# Export Excel : feuille 1 = table, feuille 2 = récap total
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    export_table.to_excel(writer, sheet_name="surfaces_par_culture", index=False)
    pd.DataFrame([{"SURFACE_TOTALE_CULTURES_ANNUELLES": surface_totale_annuelles}]) \
      .to_excel(writer, sheet_name="recap_total", index=False)



#%% Construction matrice transitions
# Construire le dictionnaire {ID_PARCEL: CODE_CULTU} pour df2
parcel_to_cultu_df2 = dict(zip(df2_annuelles["ID_PARCEL"], df2_annuelles["CODE_CULTU"]))

# Initialiser la matrice de transition
adj_matrix = defaultdict(lambda: defaultdict(float))
#%% Remplissage de la matrice
# Parcourir df1 filtré (annuelles) et remplir la matrice de transition
surf_not_matched=0
for _, row in df1_annuelles.iterrows():
    id_parcel = row["ID_PARCEL"]
    surf_parc = row["SURF_PARC"]
    cultu_2022 = row["CODE_CULTU"]  # culture ds l'année 1

    if id_parcel in parcel_to_cultu_df2:
        cultu_2023 = parcel_to_cultu_df2[id_parcel]  # culture ds l'année 2
        adj_matrix[cultu_2022][cultu_2023] += surf_parc
    else:
        # Cas 2 : la parcelle n’est pas retrouvée dans l'année 2
        surf_not_matched += surf_parc

print(f"Surface totale des parcelles annuelles de 2022 sans correspondance en 2023 : {surf_not_matched}")

# Convertir le dictionnaire en DataFrame
adj_df = pd.DataFrame.from_dict(adj_matrix, orient="index").fillna(0)

# Recréer toutes les lignes et colonnes pour avoir une matrice carrée
all_cultures = sorted(set(list(adj_df.index) + list(adj_df.columns)))
adj_df = adj_df.reindex(index=all_cultures, columns=all_cultures, fill_value=0)

# Remettre l'index comme colonne pour Excel
adj_df.reset_index(inplace=True)
adj_df.rename(columns={'index': 'CODE_CULTURE_1'}, inplace=True)



#%% Exportation resultat excel
adj_df.to_excel(
    r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\Transitions\matrice_transitions_annuelles_23_24.xlsx",
    index=False
)














#%% A STOCKER VERSION QUI FONCTIONNE
import pandas as pd
from collections import defaultdict
#%%

# Charger les deux fichiers CSV
df1 = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\RPG csv avec codegrp\rpg 2022.csv")
df2 = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\RPG csv avec codegrp\rpg 2023.csv")

# Standardiser les noms de colonnes
df1.columns = df1.columns.str.strip().str.upper()
df2.columns = df2.columns.str.strip().str.upper()

#%% tests
# Surface totale en 2022 pour la culture BTH
surf_BTH_2022 = df1[df1["CODE_CULTU"] == "BTH"]["SURF_PARC"].sum()
print(f"Surface totale en 2022 cultivée en BTH : {surf_BTH_2022} ha")

# Surface totale en 2023 pour la culture BTH
surf_BTH_2023 = df2[df2["CODE_CULTU"] == "BTH"]["SURF_PARC"].sum()
print(f"Surface totale en 2023 cultivée en BTH : {surf_BTH_2023} ha")

# Surface totale en 2022 pour la culture MIS
surf_MIS_2022 = df1[df1["CODE_CULTU"] == "MIS"]["SURF_PARC"].sum()
print(f"Surface totale en 2022 cultivée en MIS : {surf_MIS_2022} ha")

# Surface totale en 2023 pour la culture MIS
surf_MIS_2023 = df2[df2["CODE_CULTU"] == "MIS"]["SURF_PARC"].sum()
print(f"Surface totale en 2023 cultivée en MIS : {surf_MIS_2023} ha")

print(len(df1))
print(len(df2))

#%%
# Construire un dictionnaire {ID_PARCEL: CODE_CULTU} pour df2
parcel_to_cultu_df2 = dict(zip(df2["ID_PARCEL"], df2["CODE_CULTU"]))

# Initialiser la matrice de transition
adj_matrix = defaultdict(lambda: defaultdict(float))

# Parcourir df1 et remplir la matrice de transition
for _, row in df1.iterrows():
    id_parcel = row["ID_PARCEL"]
    surf_parc = row["SURF_PARC"]
    cultu_output = row["CODE_CULTU"]

    # Si la parcelle existe dans df2, on incrémente la surface dans la transition correspondante
    if id_parcel in parcel_to_cultu_df2:
        cultu_output2 = parcel_to_cultu_df2[id_parcel]
        adj_matrix[cultu_output][cultu_output2] += surf_parc

# Convertir le dictionnaire en DataFrame
adj_df = pd.DataFrame.from_dict(adj_matrix, orient="index").fillna(0)
#%%
# Exporter en CSV
adj_df.to_excel(
    r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\matrice_transitions_cultures_v2.xlsx",
    index=False
)