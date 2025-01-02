import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import math

# Read data from file crop_yield_data.csv
def read_data():
    data=np.genfromtxt('crop_yield_data.csv', delimiter=',')
    #select random 50 rows
    np.random.seed(42)
    np.random.shuffle(data)
    data=data[:50]
    return data 


# Generate synthetic data
def generate_data(n_points=20, n_features=2):
    np.random.seed(42)
    return np.random.rand(n_points, n_features)

# Compute distance matrix
def compute_distance_matrix(A):
    return squareform(pdist(A))

# A = generate_data()
A= read_data()
D = compute_distance_matrix(A)
m, n = A.shape

# print("Data matrix A:")
# print(A)
# print("Distance matrix D:")
# print(D)
# print("Shape of A:", A.shape)

def create_ampl_files(A, D, k, model_filename="cluster_median.mod", data_filename="cluster_median.dat"):
    # Create .mod file for AMPL
    model_content = """
    param m;   # Number of points
    param k;   # Number of clusters
    param D{1..m, 1..m};  # Distance matrix
    
    var x{1..m, 1..m} binary;  # Assignment variables
    
    minimize total_distance:
        sum {i in 1..m, j in 1..m} D[i, j] * x[i, j];
    
    subject to assign_cluster {i in 1..m}:
        sum {j in 1..m} x[i, j] = 1;
    
    subject to cluster_count:
        sum {j in 1..m} x[j, j] = k;
    
    subject to valid_cluster {i in 1..m, j in 1..m}:
        x[i, j] <= x[j, j];
    """
    with open(model_filename, "w") as mod_file:
        mod_file.write(model_content)
    
    # Create .dat file for AMPL
    data_content = f"""
    param m := {m};
    param k := {k};
    param D : {' '.join(map(str, range(1, m + 1)))} :=
    """
    for i, row in enumerate(D, start=1):
        data_content += f"{i} " + " ".join(map(str, row)) + "\n"
    data_content += ";"
    
    with open(data_filename, "w") as dat_file:
        dat_file.write(data_content)

create_ampl_files(A, D, k=3)


def solve_mst_heuristic(D, k):
    """
    MST-based clustering in k clusters by removing (k-1) heaviest edges from the MST.
    
    Parameters
    ----------
    D : 2D array-like
        A symmetric distance (or cost) matrix of shape (n, n).
    k : int
        Desired number of clusters.

    Returns
    -------
    clusters : list of lists
        A list of connected components, each component being a list of node indices.
    """
    n = len(D)
    
    # 1) Build an undirected graph from the distance matrix
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            w = D[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)
    
    # 2) Compute MST of this graph
    T = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
    
    # 3) Sort MST edges by descending weight
    mst_edges = sorted(T.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    # 4) Remove top (k-1) heaviest edges
    for i in range(k - 1):
        u, v, data = mst_edges[i]
        T.remove_edge(u, v)
    
    # 5) Find connected components in the pruned MST
    components = nx.connected_components(T)
    clusters = [list(comp) for comp in components]
    
    return clusters

heuristic_clusters = solve_mst_heuristic(D, k=3)
print("Heuristic Clusters:", heuristic_clusters)


def compute_cluster_median_cost(D, clusters):
    """
    Given a distance matrix D and a partition of the nodes into clusters,
    compute:
      1) The sum of distances of all nodes to their cluster 'median'.
      2) Which node is chosen as the median for each cluster.

    The 'median' of a cluster C is the node j in C that minimizes
    sum(D[i, j] for i in C).
    
    Parameters
    ----------
    D : 2D array-like (e.g., NumPy array)
        Distance (or cost) matrix of shape (n, n).
    clusters : list of lists
        A list of clusters, each being a list of node indices.

    Returns
    -------
    total_cost : float
        The sum of distances from each node to its cluster's chosen median.
    medians : list
        A list of length len(clusters), where medians[c] is the median node
        in clusters[c].
    """
    total_cost = 0
    medians = []
    
    for cluster in clusters:
        best_cluster_cost = math.inf
        best_median = None
        
        # Try each node in 'cluster' as the possible median
        for candidate in cluster:
            cost = sum(D[i, candidate] for i in cluster)
            if cost < best_cluster_cost:
                best_cluster_cost = cost
                best_median = candidate
        
        total_cost += best_cluster_cost
        medians.append(best_median)
    
    return total_cost, medians


obj_value, cluster_medians = compute_cluster_median_cost(D, heuristic_clusters)

print("Medians:", cluster_medians)
print("Objective Function (Cluster-Median Cost):", obj_value)