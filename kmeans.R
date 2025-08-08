# ---
# Title: K-Means Clustering of Football Forward Metrics
# Description: This script clusters football performance metrics for forwards
#              to identify representative features for different skill groups.
# Data Source: https://github.com/MahdiNouraie/Advanced-Football-Analytics
# ---

# 1. SETUP
# -----------------------------------------------------------------------------
# It's good practice to run install.packages() in the console, not in a script.
# If you don't have these packages, uncomment the lines below and run them once.
# install.packages("ggplot2")
# install.packages("factoextra")

# Load necessary libraries
library(ggplot2)
library(factoextra)

# Set seed for reproducibility of random processes
set.seed(26)


# 2. DATA LOADING AND PREPARATION
# -----------------------------------------------------------------------------
# Read the dataset from the source URL
data <- read.csv("https://raw.githubusercontent.com/MahdiNouraie/Advanced-Football-Analytics/refs/heads/main/data_aggregated.csv",
                 header = TRUE, check.names = FALSE)

# Filter data to include only Forwards (FW)
data_fw <- data[data$Pos == 'FW', ]

# Select and scale the numeric features for clustering (columns 8 to 120)
# The result is a data frame of scaled metrics where each row is a player.
cl_data <- data.frame(scale(data_fw[, 8:120]), check.names = FALSE)
rownames(cl_data) <- data_fw$Player # Assign player names as row names for clarity


# 3. DETERMINE OPTIMAL NUMBER OF CLUSTERS FOR VARIABLES (ELBOW METHOD)
# -----------------------------------------------------------------------------
# To cluster the metrics (variables), we first need to transpose the data.
# Now, each row represents a metric, and each column represents a player.
t_cl_data <- t(cl_data)

# Calculate the Within-Cluster Sum of Squares (WSS) for a range of k (1 to 20).
# This helps us find the "elbow point" where adding more clusters gives diminishing returns.
wss_variables <- sapply(1:20, function(k) {
  kmeans(t_cl_data, centers = k, nstart = 25, iter.max = 15)$tot.withinss
})

# Create a data frame for plotting the elbow curve
elbow_data_variables <- data.frame(
  clusters = 1:20,
  wss = wss_variables
)

# Plot the elbow curve for variable clusters
# Look for the "elbow" of the curve to determine the optimal number of clusters.
ggplot(elbow_data_variables, aes(x = clusters, y = wss)) +
  geom_point(color = "darkred", size = 3) +
  geom_line(color = "darkred", linetype = "dashed") +
  labs(
    title = "Elbow Method for Optimal k (Variable Clusters)",
    x = "Number of Clusters (k)",
    y = "Total Within-Cluster Sum of Squares (WSS)"
  ) +
  scale_x_continuous(breaks = seq(1, 20, by = 1)) +
  theme_minimal()


# 4. PERFORM K-MEANS CLUSTERING ON VARIABLES
# -----------------------------------------------------------------------------
# Based on the elbow plot, k=7 appears to be a good choice.
k <- 7

# Perform k-means clustering on the transposed (variable) data
set.seed(26) # Reset seed for consistent clustering results
kmeans_result <- kmeans(t_cl_data, centers = k, nstart = 25)


# 5. IDENTIFY REPRESENTATIVE FEATURES FOR EACH CLUSTER
# -----------------------------------------------------------------------------
# The goal is to find the single metric that best represents each cluster.
# We do this by finding the variable closest to the center (centroid) of its cluster.

# Get the cluster assignment for each variable
cluster_assignments <- data.frame(
  variable = rownames(t_cl_data),
  cluster = kmeans_result$cluster
)

# Get the coordinates of the cluster centroids
cluster_centers <- kmeans_result$centers

# Initialize a vector to store the representative variable for each cluster
representatives <- character(k)

# Loop through each cluster to find its most central variable
for (i in 1:k) {
  # Get variables belonging to the current cluster
  variables_in_cluster <- cluster_assignments$variable[cluster_assignments$cluster == i]

  # Subset the data for the current cluster's variables
  data_in_cluster <- t_cl_data[variables_in_cluster, , drop = FALSE]

  # Get the centroid for the current cluster
  centroid_for_cluster <- cluster_centers[i, ]

  # Calculate the Euclidean distance of each variable to the cluster's centroid
  distances <- apply(data_in_cluster, 1, function(variable_row) {
    sqrt(sum((variable_row - centroid_for_cluster)^2))
  })

  # The representative is the variable with the minimum distance to the centroid
  representatives[i] <- names(which.min(distances))
}

# Create a final data frame showing the representative for each cluster
representative_features <- data.frame(
  cluster = 1:k,
  representative_variable = representatives
)

# Print the final result
print("Representative Features for each Cluster:")
print(representative_features)
