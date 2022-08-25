# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("Fiscal Week", "2021")
dbutils.widgets.text("Save Path", "/dbfs/FileStore/Users/l528617@8451.com/")

# COMMAND ----------

# Install easy_hml
dbutils.library.installPyPI("easy_hml")
# this restarts the Python kernel (similar to restarting the kernel in Jupyter)
dbutils.library.restartPython()

# COMMAND ----------

# Libraries
# common python packages
import pandas as pd

# internal packages
from kpi_metrics import KPI, CustomMetric
from effodata import golden_rules, Joiner, Equality
from easy_hml import Validation, generate_hml
import logging
from seg.utils import DateType
from kayday import KrogerDate

# pyspark functions
import pyspark.sql.functions as F


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Prep

# COMMAND ----------

# Read in Data
file_path = '/dbfs/FileStore/Users/l528617@8451.com/ROOTs_Brief_UPC_List_Jan_2022_cleaned_v2.csv'
ROOTS_UPC = pd.read_csv(file_path)

# convert to spark
ROOTS_UPC = spark.createDataFrame(ROOTS_UPC)

# COMMAND ----------

kpi = KPI(use_sample_mart = False)

# COMMAND ----------

ROOTS_UPC_gtin = kpi.products.join(ROOTS_UPC, 'bas_con_upc_no', 'right')
pb_gtin = ROOTS_UPC_gtin.filter('plant_based = 1')

# COMMAND ----------

pb_gtin = pb_gtin.select(['gtin_no', 'pb_gen_category'])

# COMMAND ----------

fisc_week = dbutils.widgets.get("Fiscal Week")
seg_save_path = 'dbfs:/FileStore/Users/l528617@8451.com/PB_seg/Segmentation'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Directories for Output Files + Create Log File

# COMMAND ----------

dbutils.fs.mkdirs(seg_save_path + "/" + fisc_week)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Segmentation

# COMMAND ----------

generate_hml(
  kpi=kpi,
  save_path = seg_save_path,
  fiscal_week = fisc_week,
  products_df = pb_gtin,  
  metric1 = "sales",
  metric2 = "basket_penetration",
  products_grp_col = 'pb_gen_category'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in and Generalize Segmentation

# COMMAND ----------

consumption_scores = spark.read.parquet(seg_save_path + '/' + fisc_week + '/consumption_scores')

# COMMAND ----------

scores = consumption_scores.groupBy('ehhn').pivot('pb_gen_category').sum('consumption_score')
classification = consumption_scores.groupBy('ehhn').pivot('pb_gen_category').agg(F.first('hml_classification'))

# COMMAND ----------

scores = scores.withColumnRenamed('Dairy', 'dairy_consumption_score').withColumnRenamed('Protein','protein_consumption_score')
classification = classification.withColumnRenamed('Dairy', 'dairy_hml_classification').withColumnRenamed('Protein','protein_hml_classification')

# COMMAND ----------

# consumption scores will be given 0 if there is no score and X if no seg -- this is our not engaged group
consumption_scores = scores.join(classification, 'ehhn', 'outer').fillna(0).fillna('X')

# COMMAND ----------

consumption_scores= consumption_scores.withColumn(
  'sub_classification',
  F.when(F.col("dairy_hml_classification").isin(["H", "M"]) &
         F.col("protein_hml_classification").isin(["H","M"]), "Engaged_Dairy_Protein")
  .when(F.col("dairy_hml_classification").isin(["H","M"]) & 
       F.col("protein_hml_classification").isin(["L","X"]), "Engaged_Dairy")
  .when(F.col("dairy_hml_classification").isin(["L","X"]) &
       F.col("protein_hml_classification").isin(["H","M"]), "Engaged_Protein")
  .when((F.col("dairy_hml_classification").isin(["L","X"]) &
       F.col("protein_hml_classification").isin(["L"])) | 
        (F.col("dairy_hml_classification").isin(["L"]) &
       F.col("protein_hml_classification").isin(["L", "X"])),"Low_Engagement")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write final segmentation file to csv

# COMMAND ----------

seg_final_save_path = dbutils.widgets.get('Save Path') + "plant_based_seg_" + fisc_week + ".csv"
consumption_scores.write.csv(seg_final_save_path)
