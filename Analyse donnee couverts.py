#%%

import pandas as pd
import numpy as np
import glob
import os

# %%
#liste des codes culture avec classification CIPAN/CIE/CIMS
glossaire_cultu = pd.read_csv(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\Glossaire et classification codes cultu.csv",sep=";", encoding="utf-8")
codes_cultu = glossaire_cultu["code_culture"]

# dictionnaires : {code_culture :0/1 } classification 0/1 pour typologies de couverts
dico_cipan = glossaire_cultu.set_index("code_culture")["OK CIPAN"].to_dict()
dico_cie   = glossaire_cultu.set_index("code_culture")["OK CIE"].to_dict()
dico_cims  = glossaire_cultu.set_index("code_culture")["OK CIMS"].to_dict()
dico_biomasse1  = glossaire_cultu.set_index("code_culture")["BIOMASSE 0-1"].to_dict()
dico_biomasse2  = glossaire_cultu.set_index("code_culture")["BIOMASSE 1-4"].to_dict()

# dictionnaire : {code_culture : TYPE PARCEL} pour cultures principales
dico_typeparcel = glossaire_cultu.set_index("code_culture")["TYPE PARCEL"].to_dict()

#dictionnaire: {code_culture :0/1 } si LEGUMINEUSE pour culture principale et couverts (pour tous les codes)
dico_legumineuse = glossaire_cultu.set_index("code_culture")["LEGUMINEUSE"].to_dict()

#%%
def analyse_parcelles(csv_file):
    df = pd.read_csv(csv_file, sep=",", encoding="utf-8",dtype={"CULTURE_D1": "string", "CULTURE_D2": "string"},
    skipinitialspace=True)
    df.columns = df.columns.str.strip().str.upper()

    # Normalisation des champs utilisés pour le mapping
    # (on convertit en str pour éviter les NaN qui cassent les .str.*)
    for col in ["CODE_CULTU", "CULTURE_D1", "CULTURE_D2"]:
        if col in df.columns:
            # garder les NaN comme NaN, normaliser seulement les non-nuls
            df[col] = df[col].astype("string").str.strip().str.upper()
        else:
            df[col] = pd.Series(dtype="string")

    surfparc = df["SURF_PARC"].astype(float)

    # 1) TYPE_MAIN depuis le glossaire (TYPE PARCEL) pour les cultures principales
    df["TYPE_MAIN"] = df["CODE_CULTU"].map(dico_typeparcel).fillna("-")
    df["COUVERT_LEG"] = (
        df["CULTURE_D1"].map(dico_legumineuse).fillna(0).astype(int)
    + df["CULTURE_D2"].map(dico_legumineuse).fillna(0).astype(int)
    ).gt(0).astype(int)

    df["MAIN_LEG"] = df["CODE_CULTU"].map(dico_legumineuse).fillna(0)
    
    
    # 2) Calculs de base: surf totale, surface ayant un couvert, et part de couvert diviserifiés
    surf_tot = surfparc.sum()
    
    mask_d1 = df["CULTURE_D1"].notna() & (df["CULTURE_D1"].str.strip() != "")
    mask_d2 = df["CULTURE_D2"].notna() & (df["CULTURE_D2"].str.strip() != "")
    surf_ci = surfparc[mask_d1].sum()
    surf_diversifie = surfparc[mask_d2].sum()


    # 3) Surface totale corrigée : on exclut les types non couvrables
    surf_exclues = surfparc[
        (df["TYPE_MAIN"] != "-") &
        (df["TYPE_MAIN"] != "LEGUMINEUSE") &
        (df["TYPE_MAIN"] != "AROMA")
    ].sum()
    surf_tot_cultu = surf_tot - surf_exclues
    
    # 4) Types de cultures principales des parcelles, détail
    surf_prairie = surfparc[df["TYPE_MAIN"] == "PRAIRIE"].sum()
    surf_prairietemp = surfparc[df["TYPE_MAIN"] == "PRAIRIE TEMPORAIRE"].sum()
    surf_prairietemp_leg = surfparc[(df["MAIN_LEG"] == 1) & (df["TYPE_MAIN"] == "PRAIRIE TEMPORAIRE")].sum()

    surf_jachere = surfparc[df["TYPE_MAIN"] == "JACHERE"].sum()
    surf_arbo = surfparc[df["TYPE_MAIN"] == "ARBORICULTURE"].sum()
    surf_cperenne = surfparc[df["TYPE_MAIN"] == "CULTURE PERENNE"].sum()
    surf_aroma = surfparc[df["TYPE_MAIN"] == "AROMA"].sum()
    surf_autre = surfparc[df["TYPE_MAIN"] == "AUTRE"].sum()
    surf_legumineuse = surfparc[df["TYPE_MAIN"] == "LEGUMINEUSE"].sum()
    surf_legumineuse_tot = surfparc[df["MAIN_LEG"] == 1].sum() 
    
    # 5) Mapping CIPAN / CIE / LEGUMINEUSE / BIOMASSE pour D1 et D2
    df["CIPAN_D1"] = df["CULTURE_D1"].map(dico_cipan).fillna(0).astype(int)
    df["CIE_D1"]   = df["CULTURE_D1"].map(dico_cie).fillna(0).astype(int)
    df["LEG_D1"]  = df["CULTURE_D1"].map(dico_typeparcel).fillna("")
    df["BIOMASSE1_D1"] = df["CULTURE_D1"].map(dico_biomasse1).fillna(0).astype(int) #classification binaire
    df["BIOMASSE2_D1"] = df["CULTURE_D1"].map(dico_biomasse2).fillna(0).astype(int) #classification nuance 1-4
    
    df["CIPAN_D2"] = df["CULTURE_D2"].map(dico_cipan).fillna(0).astype(int)
    df["CIE_D2"]   = df["CULTURE_D2"].map(dico_cie).fillna(0).astype(int)
    df["LEG_D2"]  = df["CULTURE_D2"].map(dico_typeparcel).fillna("")
    df["BIOMASSE1_D2"] = df["CULTURE_D2"].map(dico_biomasse1).fillna(0).astype(int) #classification binaire
    df["BIOMASSE2_D2"] = df["CULTURE_D2"].map(dico_biomasse2).fillna(0).astype(int) #classification nuance 1-4
    
    # 6) Classification : si 2 cipan alors classer en cipan; sinon classer en type couvert recherché avec détail si contient légumineuses ou cie (doublons tolérés)
    
    df["CATEGORIE"] = np.select([(df["CIPAN_D1"] + df["CIPAN_D2"] == 2),(mask_d1 & mask_d2)&(df["CIPAN_D1"] + df["CIPAN_D2"] != 2)],["CIPAN","RECHERCHE"],default="Sans catégorie")
    
    # 6.1) Sous-catégorie "RECHERCHE avec légumineuse"
    df.loc[
        (df["CATEGORIE"] == "RECHERCHE") &
        ((df["LEG_D1"] == "LEGUMINEUSE") | (df["LEG_D2"] == "LEGUMINEUSE")),
        "CATEGORIE"
    ] = "R_AVEC_LEGUMINEUSE"

    # 6.2) Sous-catégorie "RECHERCHE CIE"
    df.loc[
        (df["CATEGORIE"] == "RECHERCHE") &
        (df["CIE_D1"] + df["CIE_D2"] == 2),
        "CATEGORIE"
    ] = "R_CIE"
    
    # 7) Catégorisation selon fort ou faible biomasse
    
    df["CATEGORIE_BIOMASSE1"] = np.where(
    (df["BIOMASSE1_D1"] + df["BIOMASSE1_D2"]) >= 1,
    "FORTE_BIOMASSE1",
    "FAIBLE_BIOMASSE1"
    )

    df["CATEGORIE_BIOMASSE2"] = np.where(
        (df["BIOMASSE2_D1"] + df["BIOMASSE2_D2"]) >= 5,
        "FORTE_BIOMASSE2",
        "FAIBLE_BIOMASSE2"
    )

    """
    df["CATEGORIE_BIOMASSE1"] = np.select(
        [(df["BIOMASSE1_D1"] + df["BIOMASSE1_D2"])>=1, (mask_d1 & mask_d2)&((df["BIOMASSE1_D1"] + df["BIOMASSE1_D2"])==0)],
        ["FORTE_BIOMASSE1","FAIBLE_BIOMASSE1"],
        default="nul_biomasse1"
    ) 
    
    df["CATEGORIE_BIOMASSE2"] = np.select(
        [(df["BIOMASSE2_D1"] + df["BIOMASSE2_D2"])>=5, (mask_d1 & mask_d2)&((df["BIOMASSE2_D1"] + df["BIOMASSE2_D2"])<5)],
        ["FORTE_BIOMASSE2","FAIBLE_BIOMASSE2"],
        default="nul_biomasse2"
    ) 
    """
    
    # 8) Sommes par catégorie
    surf_CIPAN = surfparc[df["CATEGORIE"] == "CIPAN"].sum()
    surf_couvert_recherche  = surfparc[(df["CATEGORIE"] == "RECHERCHE")|(df["CATEGORIE"] == "R_AVEC_LEGUMINEUSE")|(df["CATEGORIE"] == "R_CIE")].sum()
    surf_couvert_r_legumineuse = surfparc[df["CATEGORIE"] == "R_AVEC_LEGUMINEUSE"].sum()
    surf_CIE   = surfparc[df["CATEGORIE"] == "R_CIE"].sum()
    
    surf_couvert_legumineuse = surfparc[df["COUVERT_LEG"]==1].sum()
    surf_BIOMASSE1 = surfparc[df["CATEGORIE_BIOMASSE1"] == "FORTE_BIOMASSE1"].sum()
    surf_BIOMASSE2 = surfparc[df["CATEGORIE_BIOMASSE2"] == "FORTE_BIOMASSE2"].sum()
    
    surf_BIOMASSE1_LEG = surfparc[(df["CATEGORIE_BIOMASSE1"] == "FORTE_BIOMASSE1") & (df["COUVERT_LEG"]==1)].sum()
    surf_BIOMASSE2_LEG = surfparc[(df["CATEGORIE_BIOMASSE2"] == "FORTE_BIOMASSE2") & (df["COUVERT_LEG"]==1)].sum()
    surf_faible_BIOMASSE1_LEG = surfparc[(df["CATEGORIE_BIOMASSE1"] == "FAIBLE_BIOMASSE1") & (df["COUVERT_LEG"]==1)].sum()
    surf_faible_BIOMASSE2_LEG = surfparc[(df["CATEGORIE_BIOMASSE2"] == "FAIBLE_BIOMASSE2") & (df["COUVERT_LEG"]==1)].sum()

    # 9) Top 5 couples D1/D2 les plus fréquents ET par surface cumulée
    couples = (
        df[["CULTURE_D1", "CULTURE_D2"]]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: ";".join(sorted([row["CULTURE_D1"].strip(), row["CULTURE_D2"].strip()])),
            axis=1
        )
        .str.strip(";")
    )

    # Ajouter la colonne "COUPLE" dans le dataframe
    df["COUPLE"] = couples

    # On enlève les lignes sans couple
    df_valid = df[df["COUPLE"] != ""]
    
    """
    # --- Top par fréquence 
    top20_couples_freq = df_valid["COUPLE"].value_counts().head(20)
    """
    # --- Top par surface cumulée
    top40_couples_surface = (
        df_valid.groupby("COUPLE")["SURF_PARC"]
        .sum()
        .sort_values(ascending=False)
        .head(40)
    )


    # Préparer dictionnaire de résultats
    res = {
        "Surface totale": surf_tot,
        "Total cultures annuelles": surf_tot_cultu,
        "Total couvert": surf_ci,
        
        #catégories de cultures principales
        "Prairies" : surf_prairie,
        "Prairies temporaires" : surf_prairietemp,
        "Prairies temporaires avec légumineuse" : surf_prairietemp_leg,
        "Prairies temporaires sans légumineuse" : surf_prairietemp - surf_prairietemp_leg,        
        "Cultures pérennes" : surf_cperenne,
        "Jacheres" : surf_jachere,
        "Arboriculture" : surf_arbo,
        "Plantes aromatiques" : surf_aroma,
        "Autres" : surf_autre,
        "Légumineuse annuelle" : surf_legumineuse,
        "Totale cultures avec légumineuses" : surf_legumineuse_tot,

        #typologies de couvert
        "Total couvert diversifié": surf_diversifie,
        "Total CIPAN": surf_CIPAN,
        "Total couvert recherché ": surf_couvert_recherche,
        "Total couvert recherché avec légumineuse ": surf_couvert_r_legumineuse,
        "Total couvert recherché CIE ": surf_CIE,
        "Total couvert avec légumineuse ": surf_couvert_legumineuse,
        
        #biomasse1
        "Total couvert à faible biomasse1 ": surf_ci - surf_BIOMASSE1,        
        "Total couvert à forte biomasse1 ": surf_BIOMASSE1,
        
        "Total couvert à faible biomasse1 SANS légumineuse": surf_ci - surf_BIOMASSE1 - surf_faible_BIOMASSE1_LEG,
        "Total couvert à faible biomasse1 AVEC légumineuse ": surf_faible_BIOMASSE1_LEG,
        "Total couvert à forte biomasse1 AVEC légumineuse ": surf_BIOMASSE1_LEG,
        "Total couvert à forte biomasse1 SANS légumineuse ": surf_BIOMASSE1 - surf_BIOMASSE1_LEG,
        
        #biomasse2
        "Total couvert à forte biomasse2 ": surf_BIOMASSE2,
        "Total couvert à faible biomasse2 ": surf_ci - surf_BIOMASSE2,
        
        "Total couvert à faible biomasse2 SANS légumineuse ": surf_ci - surf_BIOMASSE2 - surf_faible_BIOMASSE2_LEG,
        "Total couvert à faible biomasse2 AVEC légumineuse ": surf_faible_BIOMASSE2_LEG,     
        "Total couvert à forte biomasse2 AVEC légumineuse ": surf_BIOMASSE2_LEG,        
        "Total couvert à forte biomasse2 SANS légumineuse ": surf_BIOMASSE2 - surf_BIOMASSE2_LEG,
        
    }
    """
    # Couples les plus fréquents en nombre d'occurence
    for i, (couple, freq) in enumerate(top20_couples_freq.items(), start=1):
        res[f"Couple {i} le plus fréquent"] = couple
        res[f"Couple {i} fréquence"] = freq
    """
    # Couples avec la plus grande surface
    for i, (couple, surf) in enumerate(top40_couples_surface.items(), start=1):
        res[f"Couple {i} plus grande surface"] = couple
    for i, (couple, surf) in enumerate(top40_couples_surface.items(), start=1):
        res[f"Couple {i} surface cumulée"] = surf


    return res


#%%
fichiers = glob.glob(r"C:\Users\suzan\OneDrive\Documents\SGPE\Couverts\export data parcelles test csv\*.csv")

resultats = {}

for fichier in fichiers:
    # Extraire l'année à partir du nom du fichier (ici, les 4 derniers chiffres avant .csv)
    nom = os.path.splitext(os.path.basename(fichier))[0]  # enlève l'extension
    annee = nom.split()[-1]  # prend le dernier "mot" = l'année
    print(annee)
    resultats[annee] = analyse_parcelles(fichier)


#%%--- Détection des codes inconnus dans CULTURE_D1, CULTURE_D2 et CODE_CULTU ---

# Ensemble des codes connus dans ton glossaire
codes_connus = set(glossaire_cultu["code_culture"].astype(str).str.strip().str.upper())

# On collecte tous les codes rencontrés dans les fichiers CSV
codes_trouves = set()

for fichier in fichiers:
    df_tmp = pd.read_csv(
        fichier,
        sep=",",
        encoding="utf-8",
        dtype={"CULTURE_D1": "string", "CULTURE_D2": "string", "CODE_CULTU": "string"},
        skipinitialspace=True
    )
    df_tmp.columns = df_tmp.columns.str.strip().str.upper()
    
    for col in ["CODE_CULTU", "CULTURE_D1", "CULTURE_D2"]:
        if col in df_tmp.columns:
            codes_trouves.update(
                df_tmp[col].dropna().astype(str).str.strip().str.upper().unique()
            )

# Codes inconnus = trouvés dans les CSV mais pas dans le glossaire
codes_inconnus = sorted(codes_trouves - codes_connus)

print("Codes inconnus détectés :", codes_inconnus)

#%%
# --- Construction du tableau final ---
tableau = {}

for annee, res in resultats.items():
    surf_tot = res["Surface totale"]
    surf_tot_cultu = res["Total cultures annuelles"]
    for indicateur, valeur in res.items():
        tableau.setdefault(indicateur, {})[annee] = valeur
            
# %%
df_res = pd.DataFrame(tableau).T
print(df_res)

# %%
df_res.to_excel(r"C:\Users\suzan\OneDrive\Documents\SGPE\resultats_comparaison.xlsx")
# %%
