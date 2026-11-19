import pandas as pd
from fuzzywuzzy import process, fuzz
import re
from pyspark.sql import SparkSession
import spacy

# === Initialiser spaCy (modèle français) ===
nlp = spacy.load("fr_core_news_sm")

# === Initialiser Spark ===
spark = SparkSession.builder \
    .appName("CorrectionVoitures") \
    .config("spark.jars", "C:\\spark1\\spark-3.5.5-bin-hadoop3\\jars\\mssql-jdbc-12.10.0.jre11.jar") \
    .getOrCreate()

# === Configuration SQL Server ===
SQL_SERVER_CONFIG = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "dbVoitures",
    "user": "myuser",
    "password": "root"
}

jdbc_url = f"jdbc:sqlserver://{SQL_SERVER_CONFIG['host']}:{SQL_SERVER_CONFIG['port']};" \
           f"databaseName={SQL_SERVER_CONFIG['database']};encrypt=false;trustServerCertificate=true"

# === Chargement des données depuis SQL Server ===
cars_db = spark.read.jdbc(url=jdbc_url, table="dbo.Cars2025", properties=SQL_SERVER_CONFIG).toPandas()
errors_db = spark.read.jdbc(url=jdbc_url, table="dbo.Cars_Error3", properties=SQL_SERVER_CONFIG).toPandas()

# === Prétraitement texte avec spaCy ===
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # garder tokens non ponctuation, non stopwords
        if not token.is_punct and not token.is_stop:
            tokens.append(token.lemma_)
    cleaned_text = " ".join(tokens)
    # Correction des abréviations courantes
    abbr_dict = {
        'bw': 'bmw', 'vw': 'volkswagen', 'v w': 'volkswagen',
        'accrd': 'accord', 'ferari': 'ferrari', 'toy': 'toyota',
        'ren': 'renault', 'peuj': 'peugeot', 'cit': 'citroën',
        'merced': 'mercedes'
    }
    for abbr, full in abbr_dict.items():
        cleaned_text = re.sub(rf'\b{abbr}\b', full, cleaned_text)
    return cleaned_text.strip()

# === Préparation des données ===
cars_db['full_name'] = cars_db['Marque'] + ' ' + cars_db['Modèle']
full_names = [preprocess_text(name) for name in cars_db['full_name'].tolist()]

brands_dict = {}
for brand in cars_db['Marque'].unique():
    cleaned_brand = preprocess_text(brand)
    brand_models = cars_db[cars_db['Marque'] == brand]['Modèle'].tolist()
    brands_dict[cleaned_brand] = [preprocess_text(m) for m in brand_models]

# === Fonction de correction ===
def correct_car_name(input_name, threshold=75):
    cleaned_input = preprocess_text(input_name)
    if cleaned_input in full_names:
        idx = full_names.index(cleaned_input)
        return cars_db.iloc[idx]['full_name'], 100
    matches = process.extractBests(cleaned_input, full_names, scorer=fuzz.token_set_ratio, score_cutoff=threshold-20, limit=5)
    if matches:
        best_match, score = max(matches, key=lambda x: x[1])
        if score >= threshold:
            idx = full_names.index(best_match)
            return cars_db.iloc[idx]['full_name'], score
    words = cleaned_input.split()
    if len(words) == 1 and len(cleaned_input) <= 4:
        brand_match, brand_score = process.extractOne(cleaned_input, brands_dict.keys())
        if brand_score > 80:
            popular_model = cars_db[cars_db['Marque'].str.lower() == brand_match]['Modèle'].value_counts().index[0]
            return f"{brand_match.title()} {popular_model}", brand_score
    if len(words) >= 2:
        brand_match, brand_score = process.extractOne(words[0], brands_dict.keys())
        if brand_score < 70:
            if 'ferrari' in cleaned_input or 'ferari' in cleaned_input:
                brand_match = 'ferrari'; brand_score = 95
            elif 'merced' in cleaned_input:
                brand_match = 'mercedes'; brand_score = 90
        if brand_score > 65:
            model_text = ' '.join(words[1:])
            model_match, model_score = process.extractOne(model_text, brands_dict[brand_match], scorer=fuzz.partial_ratio)
            if model_score < 80 and len(model_text) <= 3:
                for model in brands_dict[brand_match]:
                    if model_text in model:
                        model_match = model
                        model_score = 85
                        break
            if model_score > 60:
                return f"{brand_match.title()} {model_match.title()}", (brand_score + model_score)//2
    if matches:
        best_match, score = matches[0]
        idx = full_names.index(best_match)
        return cars_db.iloc[idx]['full_name'], score
    return "non reconnue", 0

# === Traitement ===
results = []
for idx, row in errors_db.iterrows():
    erroneous_name = row['Nom_Véhicule']
    corrected, confidence = correct_car_name(erroneous_name)
    results.append({'Nom_Erroné': erroneous_name, 'Correction': corrected, 'Confiance (%)': confidence})

# === Règles de correction spécifiques ===
correction_rules = {
    r'tota.*co': 'Toyota Corolla', r'^bw$': 'BMW Série 3',
    r'ferari': 'Ferrari', r'honda.*accrd': 'Honda Accord',
    r'peugeot.*208': 'Peugeot 208', r'renault.*clio': 'Renault Clio',
    r'volkswagen.*golf': 'Volkswagen Golf'
}

for result in results:
    for pattern, replacement in correction_rules.items():
        if re.search(pattern, result['Nom_Erroné'], re.IGNORECASE):
            result['Correction'] = replacement
            result['Confiance (%)'] = 95
            break

# === Résultats en DataFrame ===
results_df = pd.DataFrame(results)

# === Enregistrement dans SQL Server ===
df_resultats_spark = spark.createDataFrame(results_df)
df_resultats_spark.write.jdbc(
    url=jdbc_url,
    table="dbo.Corrections_Voitures",
    mode="overwrite",
    properties=SQL_SERVER_CONFIG
)

# === Affichage ===
print("Corrections terminées !")
print(f"{len(results_df)} entrées traitées")
print(results_df.head())

