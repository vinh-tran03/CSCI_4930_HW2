import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("spiral-dataset.csv", delimiter="\t", header=None, names=["X", "Y", "Cluster"])

# Define custom colors for each cluster
colors = {1: "red", 2: "blue", 3: "green"}  # Change colors as needed

# Plot each cluster with a specific color
plt.figure(figsize=(8, 8))
for cluster, color in colors.items():
    cluster_data = df[df["Cluster"] == cluster]
    plt.scatter(cluster_data["X"], cluster_data["Y"], color=color, label=f"Cluster {cluster}", alpha=0.7)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Spiral Dataset Clusters")
plt.legend()
plt.show()
