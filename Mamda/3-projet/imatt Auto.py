from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, trim, lit, when
from pyspark.sql.types import StructType, StructField, StringType
import re

# 1. Création de la session Spark avec JAR JDBC pour SQL Server
spark = SparkSession.builder \
    .appName("ImmatriculationAnalysis") \
    .config("spark.jars", "C:\\spark1\\spark-3.5.5-bin-hadoop3\\jars\\mssql-jdbc-12.10.0.jre11.jar") \
    .getOrCreate()

# ----------------------------------------------------------------
# 2. Configuration de la connexion source SQL Server
# ----------------------------------------------------------------
SQL_SERVER_CONFIG = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "MamdaImmatriculations",
    "table": "my_table",
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

# Vérification de l'existence de la colonne 'type' et ajout si nécessaire
if "type" not in df.columns:
    print("La colonne 'type' est introuvable. Elle va être créée avec une valeur par défaut.")
    df = df.withColumn("type", lit("Titre de propriété"))

# ----------------------------------------------------------------
# 3. Définition de la logique de vérification/correction
# ----------------------------------------------------------------
def verifier_et_corriger(imm):
    if imm is None:
        return "", "", ""
    imm = imm.strip()
    if imm.count('-') != 1:
        return "", "", ""
    g, d = imm.split('-')
    if not (g.isdigit() and d.isdigit()):
        return "", "", ""
    if len(g) > 2 or len(d) > 6:
        return "", "", ""
    g2 = g.zfill(2)
    d2 = d.zfill(6)
    corr = f"{g2}-{d2}"
    if corr == imm:
        return "oui", "", corr
    else:
        return "non", corr, corr

schema_res = StructType([
    StructField("valide", StringType(), False),
    StructField("correction", StringType(), True),
    StructField("final_imm", StringType(), True)
])

verif_udf = udf(verifier_et_corriger, schema_res)

# ----------------------------------------------------------------
# 4. Application de l'UDF et mise à jour des colonnes
# ----------------------------------------------------------------
processed = (
    df
    .withColumn("tmp", verif_udf(trim(col("Immatriculation"))))
    .withColumn("valide", col("tmp.valide"))
    .withColumn("correction", col("tmp.correction"))
    .withColumn("type", when(col("valide") == "", "").otherwise(col("type")))  # Vide 'type' si immatriculation invalide
    .drop("tmp")
)

# ----------------------------------------------------------------
# 5. Configuration de la connexion de destination SQL Server
# ----------------------------------------------------------------
SQL_SERVER_CONFIG_SAVE = {
    "host": "DESKTOP-5PJFL35",
    "port": "1433",
    "database": "MamdaImmatriculations",
    "table": "my_table2",
    "user": "myuser",
    "password": "root"
}

jdbc_url_save = f"jdbc:sqlserver://{SQL_SERVER_CONFIG_SAVE['host']}:{SQL_SERVER_CONFIG_SAVE['port']};" \
                f"databaseName={SQL_SERVER_CONFIG_SAVE['database']};encrypt=false;trustServerCertificate=true"

# Sauvegarde des résultats dans la table SQL Server de destination
try:
    processed.write \
        .format("jdbc") \
        .option("url", jdbc_url_save) \
        .option("dbtable", SQL_SERVER_CONFIG_SAVE["table"]) \
        .option("user", SQL_SERVER_CONFIG_SAVE["user"]) \
        .option("password", SQL_SERVER_CONFIG_SAVE["password"]) \
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
        .mode("overwrite") \
        .save()

    print("Les données ont été enregistrées avec succès dans SQL Server !")
except Exception as e:
    print("Erreur lors de l'enregistrement :", str(e))

# Arrêt de la session Spark
spark.stop()
