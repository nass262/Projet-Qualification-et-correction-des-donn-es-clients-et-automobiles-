
import spacy
from fuzzywuzzy import process
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession

# === Initialiser Spark ===
spark = SparkSession.builder \
    .appName("NormalisationAdresses") \
    .config("spark.jars", "C:\\spark1\\spark-3.5.5-bin-hadoop3\\jars\\mssql-jdbc-12.10.0.jre11.jar") \
    .getOrCreate()


# Configuration de la connexion source SQL Server
SQL_SERVER_CONFIG = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "MamdaAdresses",
    "user": "myuser",
    "password": "root"
}

jdbc_url = f"jdbc:sqlserver://{SQL_SERVER_CONFIG['host']}:{SQL_SERVER_CONFIG['port']};" \
           f"databaseName={SQL_SERVER_CONFIG['database']};encrypt=false;trustServerCertificate=true"




# === Chargement des données ===

df_adresses = spark.read.jdbc(url=jdbc_url, table="dbo.Adresse", properties=SQL_SERVER_CONFIG).toPandas()
df_villes = spark.read.jdbc(url=jdbc_url, table="dbo.villes", properties=SQL_SERVER_CONFIG).toPandas()

# === Plus------------------------------------------------

df_adresses = df_adresses.repartition(130)
df_villes = df_villes.repartition(130)


# === Préparation des données ===
df_adresses['Ancienne_Adresse'] = df_adresses['AdresseComplete']
df_villes = df_villes.dropna(subset=['NomVille'])
df_villes['NomVille'] = df_villes['NomVille'].astype(str).str.upper()
catalogue_villes = df_villes['NomVille'].tolist()
mots_cles_villes = ['CASA', 'RABAT', 'MAROC', 'TANGER', 'MARRAKECH', 'FES']

# === Chargement du modèle spaCy ===
nlp = spacy.load("fr_core_news_sm")

def get_embedding(adresse):
    if not isinstance(adresse, str):
        return np.zeros(nlp.vocab.vectors.shape[1])
    return nlp(adresse).vector

def extraire_ville_similaire(adresse, villes, seuil_similarite=0.6):
    if not isinstance(adresse, str):
        return None, 0
    doc = nlp(adresse)
    normalized_input = " ".join([token.text for token in doc])
    normalized_input = re.sub(r'\s+', ' ', normalized_input).strip()
    for mot_cle in mots_cles_villes:
        if mot_cle in normalized_input.upper():
            for ville in villes:
                if mot_cle in ville:
                    return ville.title(), 95
    for ville in villes:
        if ville in normalized_input.upper():
            return ville.title(), 100
    matches = process.extractOne(normalized_input.upper(), villes)
    if matches:
        best_match, score = matches
        if score >= seuil_similarite * 100:
            return best_match.title(), score
    return None, 0

df_adresses[['Ville', 'Confiance']] = pd.DataFrame(
    df_adresses['AdresseComplete'].apply(lambda x: extraire_ville_similaire(x, catalogue_villes)).tolist(),
    index=df_adresses.index
)

def preparer_donnees_entrainement(df):
    df_train = df.dropna(subset=['Ville']).copy()
    if len(df_train) == 0:
        return None, None, None
    X = np.array([get_embedding(adresse) for adresse in df_train['AdresseComplete']])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_train['Ville'])
    return X, y, label_encoder

def entrainer_modele(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = LogisticRegression(max_iter=1000)
    modele.fit(X_train, y_train)
    score = modele.score(X_test, y_test)
    print(f"Précision du modèle : {score:.2f}")
    return modele

def predire_villes_manquantes(df, modele, label_encoder):
    df_sans_ville = df[df['Ville'].isna()].copy()
    if len(df_sans_ville) == 0 or modele is None:
        return df
    X_sans_ville = np.array([get_embedding(adresse) for adresse in df_sans_ville['AdresseComplete']])
    if X_sans_ville.shape[0] > 0:
        predictions = modele.predict(X_sans_ville)
        probas = modele.predict_proba(X_sans_ville)
        max_probas = probas.max(axis=1) * 100
        df_sans_ville['Ville'] = label_encoder.inverse_transform(predictions)
        df_sans_ville['Confiance'] = max_probas
        return pd.concat([df.dropna(subset=['Ville']), df_sans_ville])
    return df

X, y, label_encoder = preparer_donnees_entrainement(df_adresses)
if X is not None and y is not None:
    modele = entrainer_modele(X, y)
    df_adresses = predire_villes_manquantes(df_adresses, modele, label_encoder)

def nettoyer_adresse(adresse, ville):
    if pd.notna(ville) and isinstance(adresse, str):
        adresse = re.sub(r'\b' + re.escape(ville) + r'\b', '', adresse, flags=re.IGNORECASE)
    return adresse.strip()

df_adresses['Adresse'] = df_adresses.apply(lambda row: nettoyer_adresse(row['AdresseComplete'], row['Ville']), axis=1)
df_avec_ville = df_adresses.dropna(subset=['Ville'])
df_sans_ville = df_adresses[df_adresses['Ville'].isna()].drop(columns=['Ville', 'Confiance'])

# === Sauvegarde dans SQL Server ===
df_resultats_spark = spark.createDataFrame(df_adresses[['Ancienne_Adresse', 'Adresse', 'Ville', 'Confiance']])

df_resultats_spark.write.jdbc(
    url=jdbc_url,
    table="dbo.Adresses_Nettoyees",
    mode="overwrite",
    properties=SQL_SERVER_CONFIG
)

# === Affichage final ===
print("\n📍 Résultat avec ville détectée/prévue :")
print(df_avec_ville[['Ancienne_Adresse', 'Adresse', 'Ville', 'Confiance']].head(10))
print("\n⚠️ Adresses sans ville détectée :")
print(df_sans_ville.head(10))
