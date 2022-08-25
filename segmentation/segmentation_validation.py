# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("Fiscal Week", "FY2021")

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

# Checking where gtin does not have a match to base upc
pb_gtin.filter('gtin_no is null').display()

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

log_path = '/dbfs/FileStore/Users/' + dbutils.widgets.get("User Folder") + '/PB_seg/Segmentation'  + "/" + fisc_week + "/Logger/validation_log.txt"
dbutils.fs.mkdirs(log_path)
logger = logging.getLogger(name="easy_hml")
steam_hdlr = logging.StreamHandler()
file_hdlr = logging.FileHandler(log_path, mode = 'w')
logger.addHandler(steam_hdlr)
logger.addHandler(file_hdlr)
logger.setLevel(logging.INFO)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Validation

# COMMAND ----------

pb_grp_seg_val = Validation(
  save_path = seg_save_path,
  fiscal_week = fisc_week,
  products_df = pb_gtin,
  products_grp_col = 'pb_gen_category',
  metric1 = "sales",
  metric2 = "basket_penetration"
)

# COMMAND ----------

pb_grp_seg_val.test_correlation()

# COMMAND ----------

pb_grp_seg_val.test_product_growth()

# COMMAND ----------

pb_grp_seg_val.test_units()

# COMMAND ----------

pb_grp_seg_val.test_units_dfs[("q1", "Protein")].show()
pb_grp_seg_val.test_units_dfs[("q2", "Protein")].show()
pb_grp_seg_val.test_units_dfs[("q3", "Protein")].show()
pb_grp_seg_val.test_units_dfs[("q4", "Protein")].show()

# COMMAND ----------

pb_grp_seg_val.test_percentiles()
