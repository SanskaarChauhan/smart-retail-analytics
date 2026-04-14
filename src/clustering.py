# src/clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def load_clean_data():
    df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["InvoiceDate"])
    print(f"Loaded clean data: {df.shape[0]} rows")
    return df


def build_rfm(df):
    """Build RFM table (Recency, Frequency, Monetary)."""
    print("\nBuilding RFM table...")

    snapshot_date = df["InvoiceDate"].max() + pd.DateOffset(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    print(f"RFM table created: {len(rfm)} customers")
    return rfm


def scale_rfm(rfm):
    """Scale RFM features."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/rfm_scaler.pkl")

    print("RFM scaled and scaler saved.")
    return rfm_scaled


def run_kmeans(rfm, rfm_scaled, n_clusters=4):
    """Run KMeans clustering."""
    print(f"\nRunning KMeans with k={n_clusters}...")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, rfm["Cluster"])
    print(f"Silhouette Score: {score:.4f}")

    # Label clusters by Monetary value
    cluster_order = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)

    labels = {}
    base_labels = ["High Value", "Mid Value", "Low Value", "Inactive"]

    for i, cluster_id in enumerate(cluster_order.index):
        labels[cluster_id] = base_labels[i] if i < len(base_labels) else f"Segment {i+1}"

    rfm["Segment"] = rfm["Cluster"].map(labels)

    print("\nSegment distribution:")
    print(rfm["Segment"].value_counts().to_string())

    os.makedirs("models", exist_ok=True)
    joblib.dump(km, "models/kmeans.pkl")

    print("KMeans model saved.")
    return rfm


def plot_clusters(rfm):
    """Visualize clusters."""
    os.makedirs("reports/figures", exist_ok=True)

    # Scatter plot
    plt.figure(figsize=(8, 5))
    for cluster in rfm["Cluster"].unique():
        subset = rfm[rfm["Cluster"] == cluster]
        plt.scatter(subset["Frequency"], subset["Monetary"], label=f"Cluster {cluster}", alpha=0.5)

    plt.title("Customer Segments")
    plt.xlabel("Frequency")
    plt.ylabel("Monetary (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/cluster_scatter.png", dpi=150)
    plt.show()

    # Segment distribution
    seg_counts = rfm["Segment"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(seg_counts.index, seg_counts.values)
    plt.title("Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig("reports/figures/segment_distribution.png", dpi=150)
    plt.show()


def save_rfm(rfm):
    """Save RFM segments."""
    os.makedirs("data/processed", exist_ok=True)
    rfm.to_csv("data/processed/rfm_segments.csv", index=False)
    print("RFM segments saved.")


def run_clustering():
    """Full clustering pipeline."""
    df = load_clean_data()

    rfm = build_rfm(df)
    rfm_scaled = scale_rfm(rfm)

    rfm = run_kmeans(rfm, rfm_scaled)

    plot_clusters(rfm)
    save_rfm(rfm)

    print("\nClustering complete.")


if __name__ == "__main__":
    run_clustering()