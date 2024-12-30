
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
    