import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Read data from file crop_yield_data.csv
def read_data():
    data=np.genfromtxt('crop_yield_data.csv', delimiter=',')
    #select random 50 rows
    np.random.seed(42)
    np.random.shuffle(data)
    data=data[:50]
    return data 


# Compute distance matrix
def compute_distance_matrix(A):
    return squareform(pdist(A))


A= read_data()
D = compute_distance_matrix(A)
m, n = A.shape


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


def parse_cluster_assignments(assignments_text):
    """
    Parse the provided cluster assignments from text format into a usable structure.
    
    Parameters
    ----------
    assignments_text : str
        Text containing cluster assignments in the format "Point X assigned to Cluster Y".
    
    Returns
    -------
    clusters : list of lists
        A list of clusters, where each cluster is a list of point indices.
    cluster_names : list
        List of cluster names corresponding to each cluster.
    """
    # Parse the assignments
    lines = assignments_text.strip().split("\n")
    point_to_cluster = {}
    for line in lines:
        point, cluster = map(int, line.replace("Point", "").replace("assigned to Cluster", "").split())
        point_to_cluster[point - 1] = cluster  # Convert to 0-based indexing
    
    # Group points by their cluster IDs
    cluster_dict = {}
    for point, cluster in point_to_cluster.items():
        cluster_dict.setdefault(cluster, []).append(point)
    
    clusters = list(cluster_dict.values())
    cluster_names = list(cluster_dict.keys())
    
    return clusters, cluster_names

# Provided clustering results as text
provided_clustering_text = """
Point 1 assigned to Cluster 22
Point 2 assigned to Cluster 18
Point 3 assigned to Cluster 22
Point 4 assigned to Cluster 22
Point 5 assigned to Cluster 18
Point 6 assigned to Cluster 18
Point 7 assigned to Cluster 32
Point 8 assigned to Cluster 22
Point 9 assigned to Cluster 18
Point 10 assigned to Cluster 32
Point 11 assigned to Cluster 32
Point 12 assigned to Cluster 22
Point 13 assigned to Cluster 32
Point 14 assigned to Cluster 22
Point 15 assigned to Cluster 32
Point 16 assigned to Cluster 22
Point 17 assigned to Cluster 18
Point 18 assigned to Cluster 18
Point 19 assigned to Cluster 32
Point 20 assigned to Cluster 18
Point 21 assigned to Cluster 32
Point 22 assigned to Cluster 22
Point 23 assigned to Cluster 22
Point 24 assigned to Cluster 22
Point 25 assigned to Cluster 22
Point 26 assigned to Cluster 18
Point 27 assigned to Cluster 22
Point 28 assigned to Cluster 22
Point 29 assigned to Cluster 18
Point 30 assigned to Cluster 22
Point 31 assigned to Cluster 18
Point 32 assigned to Cluster 32
Point 33 assigned to Cluster 22
Point 34 assigned to Cluster 22
Point 35 assigned to Cluster 18
Point 36 assigned to Cluster 22
Point 37 assigned to Cluster 22
Point 38 assigned to Cluster 22
Point 39 assigned to Cluster 18
Point 40 assigned to Cluster 22
Point 41 assigned to Cluster 32
Point 42 assigned to Cluster 32
Point 43 assigned to Cluster 32
Point 44 assigned to Cluster 32
Point 45 assigned to Cluster 18
Point 46 assigned to Cluster 18
Point 47 assigned to Cluster 32
Point 48 assigned to Cluster 18
Point 49 assigned to Cluster 32
Point 50 assigned to Cluster 32
"""

# Parse the provided clustering
provided_clusters, provided_cluster_names = parse_cluster_assignments(provided_clustering_text)

# Add 1 to heuristic cluster medians for proper labeling
adjusted_heuristic_cluster_medians = [median + 1 for median in cluster_medians]

def visualize_clusters_side_by_side(data, clusters1, cluster_names1, clusters2, cluster_names2):
    """
    Visualize two clustering results side by side using t-SNE and scatter plots.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Original high-dimensional data.
    clusters1 : list of lists
        A list of clusters for the first method, where each cluster is a list of indices.
    cluster_names1 : list
        List of cluster names or labels corresponding to clusters1.
    clusters2 : list of lists
        A list of clusters for the second method, where each cluster is a list of indices.
    cluster_names2 : list
        List of cluster names or labels corresponding to clusters2.
    """
    # Assign cluster labels for each method
    cluster_labels1 = np.zeros(len(data), dtype=int)
    for cluster_id, cluster in enumerate(clusters1):
        for idx in cluster:
            cluster_labels1[idx] = cluster_id
    
    cluster_labels2 = np.zeros(len(data), dtype=int)
    for cluster_id, cluster in enumerate(clusters2):
        for idx in cluster:
            cluster_labels2[idx] = cluster_id
    
    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)
    
    # Create side-by-side scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Plot the first clustering result
    for cluster_id in range(len(clusters1)):
        cluster_points = reduced_data[cluster_labels1 == cluster_id]
        axes[0].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_names1[cluster_id]}"
        )
    axes[0].set_title("t-SNE visualization of MST heuristic clusters")
    axes[0].set_xlabel("t-SNE dimension 1")
    axes[0].set_ylabel("t-SNE dimension 2")
    axes[0].legend()

    # Plot the second clustering result
    for cluster_id in range(len(clusters2)):
        cluster_points = reduced_data[cluster_labels2 == cluster_id]
        axes[1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_names2[cluster_id]}"
        )
    axes[1].set_title("t-SNE visualisation of AMPL clusters")
    axes[1].set_xlabel("t-SNE dimension 1")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Visualize both clustering results side by side
visualize_clusters_side_by_side(
    A, 
    heuristic_clusters, 
    adjusted_heuristic_cluster_medians, 
    provided_clusters, 
    provided_cluster_names
)

