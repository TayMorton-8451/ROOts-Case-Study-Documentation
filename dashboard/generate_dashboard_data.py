# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("Start Fiscal Year", "FY2021")
dbutils.widgets.text("End Fiscal Year", "FY2020")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exploring Customer Behavior in Plant Based Alternatives

# COMMAND ----------

# Libraries
# common python packages
import pandas as pd

# internal packages
from kpi_metrics import KPI, CustomMetric, get_metrics
from effodata import ACDS, golden_rules, Joiner, Equality, Sifter

# pyspark functions
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# Read in Data
file_path = '/dbfs/FileStore/Users/l528617@8451.com/ROOTs_Brief_UPC_List_Jan_2022_cleaned_v2.csv'

pb_UPC = pd.read_csv(file_path)

# convert to spark
pb_UPC = spark.createDataFrame(pb_UPC)

# COMMAND ----------

kpi = KPI(spark, use_sample_mart = False)

# COMMAND ----------

start_date = dbutils.widgets.get("Start Fiscal Year")
end_date = dbutils.widgets.get("End Fiscal Year")

# COMMAND ----------

# Aggregate to the category/quarter level
kpi_by_category_by_quarter = kpi.get_aggregate(
  start_date = start_date,
  end_date = end_date,
  
  apply_golden_rules = golden_rules(), 
  
  join_with = ['products', 'date', Joiner(pb_UPC, join_cond = Equality("bas_con_upc_no"))],
  group_by = ['pb_category', 'plant_based', 'fiscal_quarter_start_date', 'fiscal_year', 'fiscal_quarter'],
  query_filters = ['plant_based is not null'],  
  
  metrics = ['units', 'sales', 'households'],
  stratum_metric_names = True
  
)

# Aggregate to the quarter and plant based y/n level
total_by_quarter = kpi.get_aggregate(
  start_date = start_date,
  end_date = end_date,
  
  apply_golden_rules = golden_rules(), 
  
  join_with = ['products', 'date', Joiner(pb_UPC, join_cond = Equality("bas_con_upc_no"))],
  group_by = ['plant_based', 'fiscal_quarter_start_date', 'fiscal_year', 'fiscal_quarter'],
  query_filters = ['plant_based is not null'],  
  
  metrics = ['units', 'sales', 'households'],
  stratum_metric_names = True
  
)

# COMMAND ----------

kpi_by_category_by_quarter = kpi_by_category_by_quarter.withColumnRenamed('fiscal_quarter', 'fiscal_quarter_year')
kpi_by_category_by_quarter = kpi_by_category_by_quarter.withColumn("fiscal_quarter", F.regexp_extract(F.col('fiscal_quarter_year'), r'(\d$)',1))
kpi_by_category_by_quarter = kpi_by_category_by_quarter.drop('fiscal_quarter_year')


total_by_quarter = total_by_quarter.withColumnRenamed('fiscal_quarter', 'fiscal_quarter_year')
total_by_quarter = total_by_quarter.withColumn("fiscal_quarter", F.regexp_extract(F.col('fiscal_quarter_year'), r'(\d$)',1))
total_by_quarter = total_by_quarter.drop('fiscal_quarter_year')

# COMMAND ----------

# Display total units, sales, and households by category
kpi_by_category_by_quarter.display()

# COMMAND ----------

# Display total units, sales, and households by quarter
total_by_quarter.display()
