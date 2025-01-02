import numpy as np
from scipy.spatial.distance import pdist, squareform

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
print("Shape of A:", A.shape)

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


from scipy.sparse.csgraph import minimum_spanning_tree

def solve_mst_heuristic(D, k):
    mst = minimum_spanning_tree(D).toarray()
    mst[mst == 0] = np.inf  # Replace zeroes with inf for edge removal
    # Remove k-1 largest edges
    edges = np.sort(mst[mst != np.inf])[-(k-1):]
    for edge in edges:
        mst[mst == edge] = np.inf
    
    # Extract clusters from the remaining graph
    clusters = []
    visited = set()
    
    def dfs(node, cluster):
        if node in visited:
            return
        visited.add(node)
        cluster.append(node)
        for neighbor, weight in enumerate(mst[node]):
            if weight != np.inf:
                dfs(neighbor, cluster)
    
    for i in range(len(D)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    return clusters

heuristic_clusters = solve_mst_heuristic(D, k=3)
print("Heuristic Clusters:", heuristic_clusters)
