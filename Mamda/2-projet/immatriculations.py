from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType
import re

# Création de la session Spark avec le JAR JDBC pour SQL Server
spark = SparkSession.builder \
    .appName("ImmatriculationAnalysis") \
    .config("spark.jars", "C:\\spark1\\spark-3.5.5-bin-hadoop3\\jars\\mssql-jdbc-12.10.0.jre11.jar") \
    .getOrCreate()

# ------------------------------------------------------------------------------
# Configuration de la connexion source SQL Server
SQL_SERVER_CONFIG = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "immatriculations",
    "table": "mytable2",
    "user": "myuser",
    "password": "root"
}

jdbc_url = f"jdbc:sqlserver://{SQL_SERVER_CONFIG['host']}:{SQL_SERVER_CONFIG['port']};" \
           f"databaseName={SQL_SERVER_CONFIG['database']};encrypt=false;trustServerCertificate=true"

# Lecture des données depuis SQL Server
df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", SQL_SERVER_CONFIG["table"]) \
    .option("user", SQL_SERVER_CONFIG["user"]) \
    .option("password", SQL_SERVER_CONFIG["password"]) \
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
    .load()
#plus------------------------------------------------
df = df.repartition(130)
# ------------------------------------------------------------------------------
# Définition des fonctions d'aide

def est_valide_arabe(imm):
    pattern = r'^\d{5}-[ABDEH]-\d{2}$'
    return bool(re.match(pattern, imm))

def est_valide_ww(imm):
    pattern = r'^WW\s?\d{6}$'
    return bool(re.match(pattern, imm))

def corriger_format_arabe(imm):
    match = re.match(r'^(\d+)-([ABDEH])-(\d+)$', imm)
    if match:
        chiffres1, lettre, chiffres2 = match.groups()
        chiffres1_corr = chiffres1.zfill(5)
        chiffres2_corr = chiffres2.zfill(2)
        imm_corr = f"{chiffres1_corr}-{lettre}-{chiffres2_corr}"
        if est_valide_arabe(imm_corr):
            return imm_corr
    return None

def corriger_format_ww(imm):
    imm_corr = re.sub(r'[^A-Za-z0-9]', '', imm)  # Retire espaces et symboles
    match = re.match(r'^WW(\d+)$', imm_corr)
    if match:
        chiffres = match.group(1)
        chiffres_corr = chiffres.zfill(6)
        imm_corr = f"WW{chiffres_corr}"
        if est_valide_ww(imm_corr):
            return imm_corr
    return None

def contient_deux_lettres_invalides(imm):
    lettres = re.findall(r'[A-Za-z]', imm)
    invalides = [lettre for lettre in lettres if lettre.upper() not in 'ABDEH']
    return len(invalides) >= 2

# ------------------------------------------------------------------------------
# Fonction principale de traitement de l'immatriculation
def traiter_immatriculation(imm):
    imm = imm.strip() if imm else ""

    if est_valide_arabe(imm):
        return ("oui", "arabe", "")
    elif est_valide_ww(imm):
        # Pour WW, on enlève les espaces inutiles et si besoin on corrige le format
        imm_corr = re.sub(r'[^A-Za-z0-9]', '', imm)
        correction = imm_corr if imm != imm_corr else ""
        return ("oui", "WW", correction)
    else:
        # Tentative de correction pour WW
        imm_corr_ww = corriger_format_ww(imm)
        if imm_corr_ww:
            return ("non", "WW", imm_corr_ww)
        else:
            # Sinon tentative de correction pour le format arabe
            imm_corr_arabe = corriger_format_arabe(imm)
            if imm_corr_arabe and est_valide_arabe(imm_corr_arabe):
                return ("non", "arabe", imm_corr_arabe)
            elif contient_deux_lettres_invalides(imm):
                return ("", "", "")
            else:
                return ("", "", "")

# Définition du schéma de retour de la UDF
schema_retour = StructType([
    StructField("valide", StringType()),
    StructField("type", StringType()),
    StructField("correction", StringType())
])

# Enregistrement de la UDF
traiter_immatriculation_udf = udf(traiter_immatriculation, schema_retour)

# Application du traitement sur la colonne "Immatriculation"
resultat_df = df.withColumn("resultats", traiter_immatriculation_udf(col("Immatriculation"))) \
                .select(
                    "*",
                    col("resultats.valide").alias("valide"),
                    col("resultats.type").alias("type"),
                    col("resultats.correction").alias("correction")
                ).drop("resultats")

# ------------------------------------------------------------------------------
# Configuration de la connexion de destination SQL Server
SQL_SERVER_CONFIG_SAVE = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "immatriculations2",
    "table": "immatriculations_traitees",
    "user": "myuser",
    "password": "root"
}

jdbc_url_save = f"jdbc:sqlserver://{SQL_SERVER_CONFIG_SAVE['host']}:{SQL_SERVER_CONFIG_SAVE['port']};" \
                f"databaseName={SQL_SERVER_CONFIG_SAVE['database']};encrypt=false;trustServerCertificate=true"

# Sauvegarde des résultats dans la table SQL Server de destination
resultat_df.write \
        .format("jdbc") \
        .option("url", jdbc_url_save) \
        .option("dbtable", SQL_SERVER_CONFIG_SAVE["table"]) \
        .option("user", SQL_SERVER_CONFIG_SAVE["user"]) \
        .option("password", SQL_SERVER_CONFIG_SAVE["password"]) \
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
        .mode("overwrite") \
        .save()
