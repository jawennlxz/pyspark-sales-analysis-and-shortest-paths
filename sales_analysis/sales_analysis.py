"""Sales analysis with PySpark DataFrames.

Answers seven business questions on a sales dataset (sales_data.csv):
regional profit, peak 72-hour order periods, monthly demand, profit margins,
top orders per item type, year-on-year revenue growth and the top item type
in each region.

Run: spark-submit sales_analysis.py   (sales_data.csv in the same folder)
"""
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# Q2(a) Load the data; find the most profitable region and the top country in it
spark = SparkSession.builder.master("local[*]").appName("TMA_Q2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

base = os.path.dirname(os.path.abspath(__file__))
df = spark.read.csv(os.path.join(base, "sales_data.csv"),
                    header=True, inferSchema=True)
df.printSchema()
df.show(5, truncate=False)

region_profit = (df.groupBy("Region")
                 .agg(F.round(F.sum("Total Profit"), 2).alias("Total_Profit"))
                 .orderBy(F.desc("Total_Profit")))
region_profit.show(truncate=False)
top_region = region_profit.first()["Region"]
print("Region with the highest total profit:", top_region)

country_profit = (df.filter(F.col("Region") == top_region)
                  .groupBy("Country")
                  .agg(F.round(F.sum("Total Profit"), 2).alias("Total_Profit"))
                  .orderBy(F.desc("Total_Profit")))
country_profit.show(5, truncate=False)
top_country = country_profit.first()["Country"]
print("Country with the highest profit in", top_region + ":", top_country)

# Q2(b) Peak 72-hour period by number of orders (sliding window over hours)
orders = (df.withColumn("order_date", F.to_date("Order Date", "M/d/yyyy"))
          .withColumn("order_hour",
                      F.datediff("order_date", F.lit("1970-01-01").cast("date")) * 24))

w = Window.orderBy("order_hour").rangeBetween(0, 71)
windows = (orders.withColumn("orders_in_72h", F.count("Order ID").over(w))
           .select(F.col("order_date").alias("window_start"),
                   F.date_add("order_date", 2).alias("window_end"),
                   "orders_in_72h"))
windows.orderBy(F.desc("orders_in_72h"), "window_start").show(10)

max_orders = windows.agg(F.max("orders_in_72h")).first()[0]
peak = windows.filter(F.col("orders_in_72h") == max_orders).orderBy("window_start")
peak.show()
print("Highest number of orders in any 72-hour interval:", max_orders)

# Q2(c) Month with the highest average units sold per order
monthly = (orders.withColumn("month_num", F.month("order_date"))
           .withColumn("month_name", F.date_format("order_date", "MMMM"))
           .groupBy("month_num", "month_name")
           .agg(F.round(F.avg("Units Sold"), 2).alias("Avg_Units_Sold"),
                F.count("*").alias("Num_Orders"))
           .orderBy(F.desc("Avg_Units_Sold")))
monthly.drop("month_num").show(12, truncate=False)

top_month = monthly.first()
print("Month with the highest average units sold:", top_month["month_name"],
      "(average of", top_month["Avg_Units_Sold"], "units per order)")

# Q2(d) Profit margin percentage per transaction
margin_df = df.withColumn("Margin_Pct",
                          (F.col("Unit Price") - F.col("Unit Cost")) / F.col("Unit Price") * 100)
margin_df.select("Order ID", "Item Type", "Unit Price", "Unit Cost",
                 F.round("Margin_Pct", 2).alias("Margin_Pct")).show(5)

bounds = margin_df.agg(F.min("Margin_Pct").alias("lowest"),
                       F.max("Margin_Pct").alias("highest")).first()
print("Lowest margin %:", round(bounds["lowest"], 2))
print("Highest margin %:", round(bounds["highest"], 2))

for label, v in [("Lowest", bounds["lowest"]), ("Highest", bounds["highest"])]:
    print(label, "margin occurs in:")
    (margin_df.filter(F.col("Margin_Pct") == v)
     .groupBy("Item Type", "Unit Price", "Unit Cost")
     .agg(F.round(F.first("Margin_Pct"), 2).alias("Margin_Pct"),
          F.count("*").alias("Num_Orders"))
     .show(truncate=False))

(margin_df.groupBy("Item Type")
 .agg(F.round(F.min("Margin_Pct"), 2).alias("Min_Margin_Pct"),
      F.round(F.max("Margin_Pct"), 2).alias("Max_Margin_Pct"))
 .orderBy("Min_Margin_Pct").show(truncate=False))

# Q2(e) Top 3 highest-value orders for a chosen item type, and for every item type
item_type = "Cosmetics"

top3 = (df.filter(F.col("Item Type") == item_type)
        .select("Order ID", "Item Type", "Region", "Country",
                "Order Date", "Units Sold", "Total Revenue")
        .orderBy(F.desc("Total Revenue"))
        .limit(3))
print("Top 3 highest-value orders for item type:", item_type)
top3.show(truncate=False)

w = Window.partitionBy("Item Type").orderBy(F.desc("Total Revenue"))
(df.withColumn("Rank", F.row_number().over(w))
 .filter(F.col("Rank") <= 3)
 .select("Item Type", "Rank", "Order ID", "Country", "Total Revenue")
 .orderBy("Item Type", "Rank")
 .show(40, truncate=False))

# Q2(f) Annual revenue and year-on-year growth
annual = (orders.withColumn("Year", F.year("order_date"))
          .groupBy("Year")
          .agg(F.sum("Total Revenue").cast("decimal(18,2)").alias("Annual_Revenue"),
               F.count("*").alias("Num_Orders"))
          .orderBy("Year"))
annual.show()

w = Window.orderBy("Year")
growth = (annual.withColumn("Previous_Year_Revenue", F.lag("Annual_Revenue").over(w))
          .withColumn("YoY_Growth_Pct",
                      F.round((F.col("Annual_Revenue") - F.col("Previous_Year_Revenue"))
                              / F.col("Previous_Year_Revenue") * 100, 2)))
growth.select("Year", "Annual_Revenue", "Previous_Year_Revenue",
              "YoY_Growth_Pct").show(truncate=False)

# Q2(g) Top-performing item type in each region, with its global rank
region_item = (df.groupBy("Region", "Item Type")
               .agg(F.sum("Total Profit").cast("decimal(18,2)").alias("Total_Profit")))
region_item.orderBy("Region", F.desc("Total_Profit")).show(10, truncate=False)

w_region = Window.partitionBy("Region").orderBy(F.desc("Total_Profit"))
w_global = Window.orderBy(F.desc("Total_Profit"))
ranked = (region_item.withColumn("Rank_In_Region", F.rank().over(w_region))
          .withColumn("Global_Rank", F.rank().over(w_global)))

(ranked.filter(F.col("Rank_In_Region") == 1)
 .select("Region", "Item Type", "Total_Profit", "Rank_In_Region", "Global_Rank")
 .orderBy("Global_Rank")
 .show(truncate=False))

spark.stop()
