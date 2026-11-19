import pandas as pd

# Charger le DataFrame à partir du fichier CSV
df = pd.read_csv('imatt-Auto.csv')

# Afficher les colonnes existantes pour déboguer
print("Colonnes présentes dans le DataFrame :", df.columns.tolist())

# Créer la colonne "type" avec une valeur par défaut si elle n'existe pas
if 'type' not in df.columns:
    print("La colonne 'type' est introuvable. Elle va être créée avec une valeur par défaut.")
    df['type'] = "Titre de propriété"

def verifier_et_corriger(immatriculation):
    """
    Vérifie que l'immatriculation correspond à un format composé de deux parties séparées par un tiret.
    Chaque partie doit être numérique. La première partie ne doit pas comporter plus de 2 chiffres
    et la deuxième pas plus de 6 chiffres.

    En cas de non-conformité, renvoie deux valeurs vides.
    Sinon, complète avec des zéros à gauche pour atteindre 2 chiffres à gauche et 6 chiffres à droite.
    Retourne :
        - "oui" et chaîne vide si aucun ajustement n'est nécessaire,
        - "non" et le format corrigé sinon.
    """
    # Vérifier qu'il y a exactement un tiret
    if immatriculation.count('-') != 1:
        return "", ""

    partie_gauche, partie_droite = immatriculation.split('-')

    # Vérifier que les deux parties sont numériques
    if not (partie_gauche.isdigit() and partie_droite.isdigit()):
        return "", ""

    # Si la partie gauche comporte plus de 2 chiffres ou la droite plus de 6, on ignore
    if len(partie_gauche) > 2 or len(partie_droite) > 6:
        return "", ""

    # Complétion avec des zéros à gauche
    partie_gauche_corrigee = partie_gauche.zfill(2)
    partie_droite_corrigee = partie_droite.zfill(6)
    immatriculation_corrigee = f"{partie_gauche_corrigee}-{partie_droite_corrigee}"

    if immatriculation_corrigee == immatriculation:
        return "oui", ""  # format déjà correct
    else:
        return "non", immatriculation_corrigee

def traiter_ligne(row):
    """
    Applique la vérification et correction sur chaque ligne.
    Si l'immatriculation ne respecte pas les conditions, on vide les colonnes 'type' et 'correction'.
    Sinon, on renseigne 'valide' et 'correction' selon le résultat.
    """
    immatriculation = row["Immatriculation"]
    type_origine = row["type"]

    valide, correction = verifier_et_corriger(immatriculation)

    # Si l'immatriculation n'est pas traitable (valide == ""), on vide 'type' et 'correction'
    if valide == "":
        return pd.Series({"valide": "", "type": "", "correction": ""})
    else:
        # Conserver la valeur initiale de "type"
        return pd.Series({"valide": valide, "type": type_origine, "correction": correction})

# Appliquer la fonction de traitement sur chaque ligne du DataFrame
df_update = df.apply(traiter_ligne, axis=1)

# Mettre à jour (ou ajouter) les colonnes dans le DataFrame d'origine
df["valide"] = df_update["valide"]
df["type"] = df_update["type"]
df["correction"] = df_update["correction"]

# Afficher le DataFrame mis à jour
print(df)

# Sauvegarder le résultat dans un fichier Excel
df.to_excel('resultat.xlsx', index=False)
