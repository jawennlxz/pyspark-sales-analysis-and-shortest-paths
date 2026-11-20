"""Single-source shortest paths (Dijkstra) with PySpark RDDs.

Each input line describes one node: "<node_id> <initial_distance> <n1>,<w1>:<n2>,<w2>:..."
The source node starts at distance 0 and every other node at 10000 (unreached).

Every iteration updates all nodes in parallel with a flatMap (each reached node
offers distances to its neighbours) and a reduceByKey (each node keeps its best
offer). The loop stops when no distance or path changes between iterations.

Run: spark-submit dijkstra_spark.py   (graph files in the same folder)
"""
import os

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("TMA_Q3").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

INF = 10000
base = os.path.dirname(os.path.abspath(__file__))


# Q3(a) Parse the input file into (node_ID, (distance, neighbours, path))
def parse_line(line):
    parts = line.strip().split()
    node_id = int(parts[0])
    distance = int(parts[1])
    neighbors = []
    if len(parts) > 2:
        for item in parts[2].split(":"):
            if item:
                n, w = item.split(",")
                neighbors.append((int(n), int(w)))
    path = str(node_id) if distance == 0 else ""
    return (node_id, (distance, neighbors, path))


lines = sc.textFile(os.path.join(base, "5-node-graph.txt"))
print("Raw input lines:")
for l in lines.collect():
    print(l)

graph_rdd = lines.filter(lambda l: l.strip() != "").map(parse_line)

print("Parsed RDD content:")
for row in graph_rdd.sortByKey().collect():
    print(row)
print("Number of nodes:", graph_rdd.count())


# Q3(b), Q3(c) Iterative Dijkstra
def expand(record):
    """Map step: keep the node's own record and offer distances to its neighbours."""
    node, (dist, nbrs, path) = record
    out = [(node, (dist, nbrs, path))]
    if dist < INF:
        for n, w in nbrs:
            out.append((n, (dist + w, [], path + "→" + str(n))))
    return out


def keep_best(a, b):
    """Reduce step: keep the neighbour list and the shortest (distance, path) offer."""
    nbrs = a[1] if len(a[1]) > 0 else b[1]
    best = min((a[0], a[2]), (b[0], b[2]))
    return (best[0], nbrs, best[1])


def dijkstra(graph_rdd, show_steps=False):
    current = graph_rdd.cache()
    iteration = 0
    if show_steps:
        print("Initial state (before iteration 1):")
        for r in current.sortByKey().collect():
            print("  ", (r[0], (r[1][0], r[1][2])))
    while True:
        iteration += 1
        updated = current.flatMap(expand).reduceByKey(keep_best).cache()

        # Count nodes whose distance or path changed since the last iteration
        changed = (updated.join(current)
                   .filter(lambda x: x[1][0][0] != x[1][1][0] or x[1][0][2] != x[1][1][2])
                   .count())

        if show_steps:
            print(f"Iteration {iteration}: {changed} node(s) changed")
            for r in updated.sortByKey().collect():
                print("  ", (r[0], (r[1][0], r[1][2])))

        current = updated
        if changed == 0:
            break
    return current, iteration


# Q3(d) Run on the 5-node graph and print the result by node ID
final_rdd, num_iterations = dijkstra(graph_rdd, show_steps=True)
print("Number of iterations needed:", num_iterations)
print("Number of iterations that changed a distance or path:", num_iterations - 1)

sorted_result = (final_rdd.map(lambda r: (r[0], (r[1][0], r[1][2])))
                 .sortByKey(ascending=True)
                 .collect())
print("Final output sorted by ascending node_ID:")
print(sorted_result)


# Q3(e) Furthest nodes, nodes with the most hops, unreachable nodes
def analyse(final_rdd):
    result_rdd = final_rdd.map(lambda r: (r[0], (r[1][0], r[1][2])))
    reachable = result_rdd.filter(lambda r: r[1][0] < INF)  # 10000 means not reached

    top3 = reachable.sortBy(lambda r: (-r[1][0], r[0])).take(3)
    print("Top 3 furthermost nodes (node, (distance, path)):")
    for r in top3:
        print("  ", r)

    hops_rdd = (reachable.filter(lambda r: r[0] != 1)
                .map(lambda r: (r[0], (r[1][0], r[1][1], len(r[1][1].split("→")) - 1))))
    max_hops = hops_rdd.map(lambda r: r[1][2]).max()
    most_hops = hops_rdd.filter(lambda r: r[1][2] == max_hops).sortByKey().collect()
    print("Maximum number of hops:", max_hops)
    print("Destination node(s) with the most hops (node, (distance, path, hops)):")
    for r in most_hops:
        print("  ", r)

    unreachable = (result_rdd.filter(lambda r: r[1][0] >= INF)
                   .map(lambda r: r[0]).sortBy(lambda x: x).collect())
    print("Nodes not reachable from node 1:", unreachable if unreachable else "None")
    return result_rdd, hops_rdd


result_rdd, hops_rdd = analyse(final_rdd)


# Q3(f) Repeat for the 20-node and 40-node graphs
def run_graph(filename):
    print("=" * 60)
    print("Graph file:", filename)
    g = (sc.textFile(os.path.join(base, filename))
         .filter(lambda l: l.strip() != "")
         .map(parse_line))
    fin, its = dijkstra(g)

    print("Number of iterations needed:", its)
    print("Number of iterations that changed a distance or path:", its - 1)
    sorted_result = (fin.map(lambda r: (r[0], (r[1][0], r[1][2])))
                     .sortByKey(ascending=True).collect())
    print("Final output sorted by ascending node_ID:")
    for row in sorted_result:
        print("  ", row)

    analyse(fin)
    return fin


result_20 = run_graph("20-node-graph.txt")
result_40 = run_graph("40-node-graph.txt")


# Q3(g) Shortest path tree of the 20-node graph: nodes per hop level
reachable_20 = (result_20.map(lambda r: (r[0], (r[1][0], r[1][2])))
                .filter(lambda r: r[1][0] < INF))
levels = reachable_20.map(lambda r: (len(r[1][1].split("→")) - 1, (r[0], r[1][0])))

print("Nodes in each hop level (node, shortest distance):")
for lvl, nodes in levels.groupByKey().mapValues(lambda v: sorted(v)).sortByKey().collect():
    print("  ", lvl, nodes)

hop_summary = (levels.map(lambda x: (x[0], (1, x[1][1])))
               .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
               .map(lambda x: (x[0], x[1][0], round(x[1][1] / x[1][0], 2)))
               .sortBy(lambda x: x[0])
               .collect())

print("(hop level, number of nodes, average shortest distance):")
for row in hop_summary:
    print("  ", row)


# Q3(h) Average edge weight per destination node = distance / hop count
edge_rdd = (reachable_20.map(lambda r: (r[0], r[1][0], len(r[1][1].split("→")) - 1))
            .map(lambda r: (r[0], r[1], r[2], 0.0 if r[2] == 0 else r[1] / r[2])))

print("All reachable nodes (node ID, shortest distance, hop count, average edge weight):")
for r in edge_rdd.sortBy(lambda r: r[0]).collect():
    print("  ", (r[0], r[1], r[2], round(r[3], 2)))

largest = edge_rdd.sortBy(lambda r: (-r[3], r[0])).take(3)
print("Top 3 largest average edge weight (node ID, shortest distance, hop count, average edge weight):")
for r in largest:
    print("  ", (r[0], r[1], r[2], round(r[3], 2)))

nonzero = edge_rdd.filter(lambda r: r[3] > 0)
smallest = nonzero.sortBy(lambda r: (r[3], r[0])).take(3)
print("Top 3 smallest non-zero average edge weight (node ID, shortest distance, hop count, average edge weight):")
for r in smallest:
    print("  ", (r[0], r[1], r[2], round(r[3], 2)))

top_ids = [r[0] for r in smallest]
ties = nonzero.filter(lambda r: r[3] == smallest[-1][3] and r[0] not in top_ids).collect()
for r in ties:
    print("   Tied with 3rd place:", (r[0], r[1], r[2], round(r[3], 2)))

spark.stop()
