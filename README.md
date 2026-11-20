# PySpark Sales Analysis and Shortest Paths

Two Apache Spark programs written in PySpark: a sales analysis that answers seven business questions with DataFrames and window functions, and a parallel version of Dijkstra's shortest-path algorithm built from RDD map and reduce steps.

Module: ICT337 Big Data Computing in the Cloud, Singapore University of Social Sciences (2026)

## Contents

| Folder | Program | What it does |
|---|---|---|
| `sales_analysis` | `sales_analysis.py` | Profit by region and country, peak 72-hour order window, monthly demand, profit margins, top orders per item type, year-on-year growth and top item type per region |
| `shortest_path_dijkstra` | `dijkstra_spark.py` | Iterative Dijkstra on 5, 20 and 40-node weighted graphs, plus analysis of the resulting shortest path tree |

Each folder also has a `sample_output.txt` with the full output of a run.

## 1. Sales analysis (DataFrames)

| Question | Approach | Result |
|---|---|---|
| Most profitable region and country | `groupBy` + `sum` | Sub-Saharan Africa; Djibouti leads within it |
| Busiest 72-hour period | Sliding window: `Window.orderBy(hour).rangeBetween(0, 71)` | At most 2 orders in any 72 hours, reached by 8 windows |
| Month with the highest average units sold | `month()` + `avg` | July, 6,350 units per order |
| Lowest and highest profit margin | `(price - cost) / price` | 13.56% (Meat) to 67.20%; margin is fixed per item type |
| Top 3 orders per item type | `row_number()` over `partitionBy("Item Type")` | Works for any item type |
| Year-on-year revenue growth | `lag()` over `orderBy("Year")` | -41.99% in 2011, +186.62% in 2012, then four years of decline |
| Top item type per region | `rank()` within region and globally | Cosmetics leads in 3 regions |

## 2. Shortest paths with Dijkstra (RDDs)

The classic algorithm settles one node at a time, which does not parallelise well. This version updates every node in each iteration:

1. **Map (`flatMap`)**: every reached node keeps its own record and offers `distance + weight` (with the extended path) to each neighbour.
2. **Reduce (`reduceByKey`)**: each node keeps its neighbour list and the shortest offer it received.
3. **Stop**: the new RDD is joined with the previous one, and the loop ends when no distance or path has changed.

| Graph | Iterations | Max hops | Unreachable nodes |
|---|---|---|---|
| 5 nodes | 4 | 3 | None |
| 20 nodes | 5 | 4 | 9 nodes |
| 40 nodes | 7 | 6 | 19, 32, 35 |

The number of iterations is always the maximum hop count plus one: information spreads one edge per iteration, and one final pass confirms nothing changed.

The program also groups the 20-node graph's shortest path tree by hop level and ranks destinations by average edge weight (distance ÷ hops).

### Input format

One line per node: `<node_id> <initial_distance> <neighbour>,<weight>:<neighbour>,<weight>:...`
The source node starts at 0 and all other nodes at 10000 (unreached), for example:

```
1 0 2,10:3,5
2 10000 3,2:4,1
```

## How to run

You need Java and Python 3.

```bash
pip install pyspark
spark-submit sales_analysis/sales_analysis.py
spark-submit shortest_path_dijkstra/dijkstra_spark.py
```

The scripts expect `sales_data.csv` and the graph files (`5-node-graph.txt`, `20-node-graph.txt`, `40-node-graph.txt`) in their own folders. These were provided for coursework and are not included.

## Skills demonstrated

- PySpark DataFrames: grouping, aggregation, date handling and window functions (`rank`, `row_number`, `lag`, range windows)
- PySpark RDDs: `map`, `flatMap`, `reduceByKey`, `join`, caching and iterative algorithms
- Designing a graph algorithm for parallel execution

## Author

Jawenn Lim
