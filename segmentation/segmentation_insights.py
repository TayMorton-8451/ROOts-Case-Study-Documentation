# Databricks notebook source
# Libraries
# common python packages
import pandas as pd

# internal packages
from kpi_metrics import KPI, CustomMetric
from effodata import golden_rules, Joiner, Equality
import seg
from seg.utils import DateType

# pyspark functions
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Reading in PB Segmentation

# COMMAND ----------

# Read in Data
file_path = '/dbfs/FileStore/Users/l528617@8451.com/ROOTs_Brief_UPC_List_Jan_2022_cleaned_v2.csv'
ROOTS_UPC = pd.read_csv(file_path)
# convert to spark
ROOTS_UPC = spark.createDataFrame(ROOTS_UPC)

# COMMAND ----------

ROOTS_UPC.display()

# COMMAND ----------

pb_grp_path = 'dbfs:/FileStore/Users/l528617@8451.com/PB_seg/Segmentation'
fisc_week = 'FY2021'

# COMMAND ----------

consumption_scores = spark.read.parquet(pb_grp_path + '/' + fisc_week + '/consumption_scores')

# COMMAND ----------

scores = consumption_scores.groupBy('ehhn').pivot('pb_gen_category').sum('consumption_score')
classification = consumption_scores.groupBy('ehhn').pivot('pb_gen_category').agg(F.first('hml_classification'))

# COMMAND ----------

scores = scores.withColumnRenamed('Dairy', 'dairy_consumption_score').withColumnRenamed('Protein','protein_consumption_score')
classification = classification.withColumnRenamed('Dairy', 'dairy_hml_classification').withColumnRenamed('Protein','protein_hml_classification')

# COMMAND ----------

scores.display()

# COMMAND ----------

# consumption scores will be given 0 if there is no score and X if no seg -- this is our not engaged group
consumption_scores = scores.join(classification, 'ehhn', 'outer').fillna(0).fillna('X')

# COMMAND ----------

consumption_scores.stat.corr('protein_consumption_score','dairy_consumption_score')

# COMMAND ----------

consumption_scores.crosstab('dairy_hml_classification','protein_hml_classification').display()

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

consumption_scores.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## HH Counts by Segmentation

# COMMAND ----------

# Use to answer how many households fall in the not engaged category for both
kpi = KPI(use_sample_mart = False)

# COMMAND ----------

start_date = 'FY2021'
end_date  = 'FY2021'

# COMMAND ----------

# Get households for FY2021 (not sure of a better way)
total_counts = kpi.get_aggregate(
  start_date = start_date,
  end_date = end_date,
  
  apply_golden_rules = golden_rules(), 
  
  group_by = ['ehhn'],
  
  metrics = 'households'
)

# COMMAND ----------

# Outer join to maintain all households
total_counts = total_counts.join(consumption_scores, 'ehhn', 'outer')

# COMMAND ----------

total_counts.display()

# COMMAND ----------

total_counts.crosstab('dairy_hml_classification','protein_hml_classification').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PB Segmentation + HH Level Data

# COMMAND ----------

kpi = KPI(use_sample_mart = True)

# COMMAND ----------

my_metrics = [
  CustomMetric('pb_sales', 'sum(case when plant_based = 1 then net_spend_amt else 0 end)'),
  CustomMetric('cv_sales', 'sum(case when plant_based = 0 then net_spend_amt else 0 end)'),
  CustomMetric('pb_units', 'sum(case when plant_based = 1 then scn_unt_qy else 0 end)'),
  CustomMetric('cv_units', 'sum(case when plant_based = 0 then scn_unt_qy else 0 end)'),
  CustomMetric(
    'pb_dairy_sales',
    'sum(case when plant_based = 1 and pb_gen_category = "Dairy" then net_spend_amt else 0 end)'
  ),
  CustomMetric(
    'cv_dairy_sales', 
    'sum(case when plant_based = 0 and pb_gen_category = "Dairy" then net_spend_amt else 0 end)'
  ),
  CustomMetric(
    'pb_protein_sales', 
    'sum(case when plant_based = 1 and pb_gen_category = "Protein" then net_spend_amt else 0 end)'
  ),
  CustomMetric(
    'cv_protein_sales', 
    'sum(case when plant_based = 0 and pb_gen_category = "Protein" then net_spend_amt else 0 end)'
  ),
  CustomMetric(
    'pb_sales_perc', 
    'sum(case when plant_based = 1 then net_spend_amt else 0 end) / sum(net_spend_amt)'
  ),
  'sales',
  'units',
]

# COMMAND ----------

# Aggregation of transactions -- all transactions are included (including upcs not in our list)
hh_agg_gen = kpi.get_aggregate(
  start_date = start_date,
  end_date = end_date,
  
  apply_golden_rules = golden_rules(), 
  
  join_with = [
    'products',
    Joiner(ROOTS_UPC, join_cond = Equality('bas_con_upc_no'))
  ],
  group_by = ['ehhn'],
  
  metrics = my_metrics,
)

# COMMAND ----------

# consumption scores will be given 0 if there is no score and X if no seg -- this is our not engaged group
hh_agg_gen_scores = hh_agg_gen.join(consumption_scores, "ehhn", "left").fillna('No_Engagement', 'sub_classification').fillna(0).fillna('X')

# COMMAND ----------

# Get segmentation info to compare with pb segmentation
hh_agg_gen_scores_w_segs = seg.get_segs_and_join(
  segs_to_join = ["funlo","st_targets", "cds_hh", "aiq.[hoh_gender, hoh_income, household_size, hoh_generation, hoh_age, ethnicity]"],
  date = end_date,
  df = hh_agg_gen_scores,
  join_type = "left"
)
# Cust 360 stopped working?
# hh_agg_gen_scores_w_segs = hh_agg_gen_scores_w_segs.join(seg.get_seg_for_date('cust_360_dietary', end_date), 'ehhn', 'left')

# COMMAND ----------

hh_agg_gen_scores_w_segs = hh_agg_gen_scores_w_segs.join(seg.get_seg_for_date('cust_360_dietary', end_date), 'ehhn', 'left')

# COMMAND ----------

# This df will be used for visuals & comparisons
hh_agg_gen_scores_w_segs.display()

# COMMAND ----------

seg_table = hh_agg_gen_scores_w_segs.groupBy('sub_classification').agg(
  F.count_distinct('ehhn').alias('households'),
  F.sum('pb_sales').alias('total_pb_sales'),
  F.sum('cv_sales').alias('total_cv_sales'),
  F.sum('pb_dairy_sales').alias('total_pb_dairy_sales'),
  F.sum('cv_dairy_sales').alias('total_cv_dairy_sales'),
  F.sum('pb_protein_sales').alias('total_pb_protein_sales'),
  F.sum('cv_protein_sales').alias('total_cv_protein_sales'), 
  F.sum('sales').alias('total_sales'),
  F.sum('units').alias('total_units'),
  F.count('sales').alias('total_sales'),
  F.sum('units').alias('total_units')
)
seg_table = seg_table \
  .withColumn('prop_pb_sales', F.col('total_pb_sales')/F.col('total_sales')) \
  .withColumn('prop_cv_sales', F.col('total_cv_sales')/F.col('total_sales')) \
  .withColumn('pb_sales_per_hh', F.col('total_pb_sales')/F.col('households')) \
  .withColumn('cv_sales_per_hh', F.col('total_cv_sales')/F.col('households')) \
  .withColumn('pb_dairy_sales_per_hh', F.col('total_pb_dairy_sales')/F.col('households')) \
  .withColumn('pb_protein_sales_per_hh', F.col('total_pb_protein_sales')/F.col('households')) \
  .withColumn('total_sales_per_hh', F.col('total_sales')/F.col('households')) 


# COMMAND ----------

dairy_table = hh_agg_gen_scores_w_segs.groupBy('dairy_hml_classification').agg(
  F.count_distinct('ehhn').alias('households'),
  F.sum('pb_sales').alias('total_pb_sales'),
  F.sum('cv_sales').alias('total_cv_sales'),
  F.sum('pb_dairy_sales').alias('total_pb_dairy_sales'),
  F.sum('cv_dairy_sales').alias('total_cv_dairy_sales'),
  F.sum('pb_protein_sales').alias('total_pb_protein_sales'),
  F.sum('cv_protein_sales').alias('total_cv_protein_sales'), 
  F.sum('sales').alias('total_sales'),
  F.sum('units').alias('total_units')
)
dairy_table = dairy_table \
  .withColumn('prop_pb_sales', F.col('total_pb_sales')/F.col('total_sales')) \
  .withColumn('prop_cv_sales', F.col('total_cv_sales')/F.col('total_sales')) \
  .withColumn('pb_sales_per_hh', F.col('total_pb_sales')/F.col('households')) \
  .withColumn('cv_sales_per_hh', F.col('total_cv_sales')/F.col('households')) \
  .withColumn('pb_dairy_sales_per_hh', F.col('total_pb_dairy_sales')/F.col('households')) \
  .withColumn('pb_protein_sales_per_hh', F.col('total_pb_protein_sales')/F.col('households')) \
  .withColumn('total_sales_per_hh', F.col('total_sales')/F.col('households')) 


# COMMAND ----------

protein_table = hh_agg_gen_scores_w_segs.groupBy('protein_hml_classification').agg(
  F.count_distinct('ehhn').alias('households'),
  F.sum('pb_sales').alias('total_pb_sales'),
  F.sum('cv_sales').alias('total_cv_sales'),
  F.sum('pb_dairy_sales').alias('total_pb_dairy_sales'),
  F.sum('cv_dairy_sales').alias('total_cv_dairy_sales'),
  F.sum('pb_protein_sales').alias('total_pb_protein_sales'),
  F.sum('cv_protein_sales').alias('total_cv_protein_sales'), 
  F.sum('sales').alias('total_sales'),
  F.sum('units').alias('total_units')
)
protein_table = protein_table \
  .withColumn('prop_pb_sales', F.col('total_pb_sales')/F.col('total_sales')) \
  .withColumn('prop_cv_sales', F.col('total_cv_sales')/F.col('total_sales')) \
  .withColumn('pb_sales_per_hh', F.col('total_pb_sales')/F.col('households')) \
  .withColumn('cv_sales_per_hh', F.col('total_cv_sales')/F.col('households')) \
  .withColumn('pb_dairy_sales_per_hh', F.col('total_pb_dairy_sales')/F.col('households')) \
  .withColumn('pb_protein_sales_per_hh', F.col('total_pb_protein_sales')/F.col('households')) \
  .withColumn('total_sales_per_hh', F.col('total_sales')/F.col('households')) 


# COMMAND ----------

seg_table.display()

# COMMAND ----------

dairy_table.display()

# COMMAND ----------

protein_table.display()

# COMMAND ----------

# hh_agg_gen_scores_w_segs.groupBy(grouping_vars).agg(
#     hh_distinct_count = pd.NamedAgg(column = "ehhn", aggfunc = "nunique"),
#     total_pb_sales = pd.NamedAgg(column = "pb_sales", aggfunc = "sum"),
#     total_cv_sales = pd.NamedAgg(column = "cv_sales", aggfunc = "sum"),
#     total_pb_dairy_sales = pd.NamedAgg(column = "pb_dairy_sales", aggfunc = "sum"),
#     total_cv_dairy_sales = pd.NamedAgg(column = "cv_dairy_sales", aggfunc = "sum"),
#     total_pb_protein_sales = pd.NamedAgg(column = "pb_protein_sales", aggfunc = "sum"),
#     total_cv_protein_sales = pd.NamedAgg(column = "cv_protein_sales", aggfunc = "sum"),
#     total_sales = pd.NamedAgg(column = "sales", aggfunc = "sum"),
#     total_units = pd.NamedAgg(column = "units", aggfunc = "sum")
#   )
#   pd_df['prop_pb_sales'] = pd_df['total_pb_sales']/pd_df['total_sales']
#   pd_df['prop_pb_units'] = pd_df['total_pb_units']/pd_df['total_units']
#   pd_df['prop_cv_sales'] = pd_df['total_cv_sales']/pd_df['total_sales']
#   pd_df['prop_cv_units'] = pd_df['total_cv_units']/pd_df['total_units']
#   pd_df['pb_sales_per_hh'] = pd_df['total_pb_sales']/pd_df['hh_distinct_count'] 
#   pd_df['cv_sales_per_hh'] = pd_df['total_cv_sales']/pd_df['hh_distinct_count']
#   pd_df['pb_dairy_sales_per_hh'] = pd_df['total_pb_dairy_sales']/pd_df['hh_distinct_count'] 
#   pd_df['pb_protein_sales_per_hh'] = pd_df['total_pb_protein_sales']/pd_df['hh_distinct_count'] 
#   pd_df['total_sales_per_hh'] = pd_df['total_sales']/pd_df['hh_distinct_count']
#   pd_df['total_units_per_hh'] = pd_df['total_units']/pd_df['hh_distinct_count']

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Extra Levels of Aggregation
# MAGIC Below is not currently being used for case -- A bit too granular for our interest

# COMMAND ----------

# Aggregation of transactions for transactions that have the UPCs of interest -- Not all transactions are included
hh_agg = kpi.get_aggregate(
  start_date = start_date,
  end_date = end_date,
  
  apply_golden_rules = golden_rules(), 
  
  join_with = [
    'products',
    Joiner(ROOTS_UPC, join_cond = Equality('bas_con_upc_no'))
  ],
  group_by = ['ehhn','pb_gen_category', 'pb_category'],
  query_filters = ['plant_based is not null'],  
  
  metrics = my_metrics,
)

# COMMAND ----------

# Joining to get all households including those excluded from the segmentation
# Note: hh is not a unique identifier, so the scores will be repeated -- future calculations need to take this into consideration
hh_agg_scores = hh_agg.join(consumption_scores, "ehhn", "left")

# COMMAND ----------

hh_agg_scores = hh_agg_scores.fillna('No_Engagement', 'sub_classification').fillna(0).fillna('X')

# COMMAND ----------

hh_agg_scores_pd = hh_agg_scores.toPandas()

# COMMAND ----------

hh_agg_scores_pd

# COMMAND ----------

def seg_metrics_grouped(pd_df, grouping_vars = []):
  pd_df = pd_df.groupby(grouping_vars).agg(
    hh_distinct_count = pd.NamedAgg(column = "ehhn", aggfunc = "nunique"),
    total_pb_sales = pd.NamedAgg(column = "pb_sales", aggfunc = "sum"),
    total_cv_sales = pd.NamedAgg(column = "cv_sales", aggfunc = "sum"),
    total_pb_units = pd.NamedAgg(column = "pb_units", aggfunc = "sum"),
    total_cv_units = pd.NamedAgg(column = "cv_units", aggfunc = "sum"),
    total_sales = pd.NamedAgg(column = "sales", aggfunc = "sum"),
    total_units = pd.NamedAgg(column = "units", aggfunc = "sum")
  )
  pd_df['prop_pb_sales'] = pd_df['total_pb_sales']/pd_df['total_sales']
  pd_df['prop_pb_units'] = pd_df['total_pb_units']/pd_df['total_units']
  pd_df['prop_cv_sales'] = pd_df['total_cv_sales']/pd_df['total_sales']
  pd_df['prop_cv_units'] = pd_df['total_cv_units']/pd_df['total_units']
  pd_df['pb_sales_per_hh'] = pd_df['total_pb_sales']/pd_df['hh_distinct_count'] 
  pd_df['pb_units_per_hh'] = pd_df['total_pb_units']/pd_df['hh_distinct_count']
  pd_df['cv_sales_per_hh'] = pd_df['total_cv_sales']/pd_df['hh_distinct_count'] 
  pd_df['cv_units_per_hh'] = pd_df['total_cv_units']/pd_df['hh_distinct_count']
  pd_df['total_sales_per_hh'] = pd_df['total_sales']/pd_df['hh_distinct_count']
  pd_df['total_units_per_hh'] = pd_df['total_units']/pd_df['hh_distinct_count']

  return pd_df


# COMMAND ----------

seg_metrics_grouped(hh_agg_scores_pd, ['sub_classification', 'pb_category']).reset_index().display()

# COMMAND ----------

seg_metrics_grouped(hh_agg_scores_pd, ['sub_classification', 'pb_gen_category']).reset_index().display()

# COMMAND ----------

seg_metrics_grouped(hh_agg_scores_pd, ['sub_classification']).reset_index().display()

# COMMAND ----------

seg_metrics_grouped(hh_agg_scores_pd, ['dairy_hml_classification']).reset_index().display()

# COMMAND ----------

seg_metrics_grouped(hh_agg_scores_pd, ['protein_hml_classification']).reset_index().display()

# COMMAND ----------

hh_agg_scores_pd.groupby(["dairy_hml_classification","protein_hml_classification"]).agg(
    hh_distinct_count = pd.NamedAgg(column = "ehhn", aggfunc = "nunique"),
)

# COMMAND ----------

date = "FY2021"
funlo = seg.get_seg_for_date("funlo",date)
st_targets = seg.get_seg_for_date("st_targets", date)
cust_360_dietary = seg.get_seg_for_date("cust_360_dietary", date)
cds = seg.get_seg_for_date("cds_hh", date)
aiq = seg.get_seg_for_date("aiq", date)

# COMMAND ----------

seg.list_available_segs()

# COMMAND ----------

def get_row_percents(cntg_tbl):
  cols = cntg_tbl.columns[1:]
  final_tbl = cntg_tbl.withColumn(
      'row_total',
      sum([F.col(c) for c in cols])
  )
  for c in cols:
    final_tbl = final_tbl.withColumn(
      c + '_rp',
      F.col(c)/F.col('row_total')
    )
  return final_tbl

# COMMAND ----------

tbl = consumption_scores.join(funlo, 'ehhn', 'left').crosstab('sub_classification','funlo_rollup_desc')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(st_targets, 'ehhn', 'left').crosstab('sub_classification','starget')
get_row_percents(tbl).display()

# COMMAND ----------

cds.limit(10).display()

# COMMAND ----------

tbl = consumption_scores.join(cust_360_dietary, 'ehhn', 'left').crosstab('sub_classification','vegan_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cust_360_dietary, 'ehhn', 'left').crosstab('sub_classification','paleo_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cust_360_dietary, 'ehhn', 'left').crosstab('sub_classification','keto_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cds, 'ehhn', 'left').crosstab('sub_classification','convenience_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cds, 'ehhn', 'left').crosstab('sub_classification','quality_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cds, 'ehhn', 'left').crosstab('sub_classification','price_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cds, 'ehhn', 'left').crosstab('sub_classification','health_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(cds, 'ehhn', 'left').crosstab('sub_classification','inspiration_dim_seg')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(st_targets, 'ehhn', 'left').crosstab('dairy_hml_classification','starget')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(st_targets, 'ehhn', 'left').crosstab('protein_hml_classification','starget')
get_row_percents(tbl).display()

# COMMAND ----------

# aiq.limit(10).display()

# COMMAND ----------

# aiq.crosstab('aiq_gender','hoh_gender').display()
# aiq.crosstab('hoh_age','hoh_legal_age').display()
# aiq.crosstab('ethniciq','ethnicity').display()

# COMMAND ----------

# hoh_income
# hoh_gender
# hoh_age
# ethnicity
# dog_flag
# cat_flag
# baby_flag
# kig_flag

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','hoh_gender')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','hoh_age')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','ethnicity')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','aiq_education')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','hoh_income')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','kids_flag')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('sub_classification','dog_flag')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('protein_hml_classification','hoh_income')
get_row_percents(tbl).display()

# COMMAND ----------

tbl = consumption_scores.join(aiq, 'ehhn', 'left').crosstab('dairy_hml_classification','hoh_income')
get_row_percents(tbl).display()
