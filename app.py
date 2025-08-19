from fastapi import FastAPI
from fastapi.responses import Response
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import PCA
import base64


app = FastAPI()

# Simulated sales dataset
sales: List[Dict[str, Any]] = [
    {"sale_id": 1, "customer_id": 101, "date": date(2025, 1, 5),  "amount": 120.0, "product": "Laptop", "channel": "Online"},
    {"sale_id": 2, "customer_id": 101, "date": date(2025, 2, 10), "amount": 80.0,  "product": "Mouse", "channel": "Store"},
    {"sale_id": 3, "customer_id": 102, "date": date(2025, 1, 15), "amount": 200.0, "product": "Monitor", "channel": "Online"},
    {"sale_id": 4, "customer_id": 103, "date": date(2025, 2, 20), "amount": 50.0,  "product": "Keyboard", "channel": "Store"},
    {"sale_id": 5, "customer_id": 102, "date": date(2025, 3, 1),  "amount": 150.0, "product": "Chair", "channel": "Online"},
    {"sale_id": 6, "customer_id": 104, "date": date(2025, 2, 28), "amount": 300.0, "product": "Desk", "channel": "Online"},
]


@app.get("/rfm")
def get_rfm() -> Dict[str, Any]:
    """
    Calculate the RFM (Recency, Frequency, Monetary) table from the sales dataset.

    Returns:
        dict: Contains the reference date and a list of customers with their RFM metrics.
    """
    if not sales:
        return {"reference_date": None, "data": [], "explanation": "No sales data available."}

    # Reference date = day after the most recent sale
    reference_date = max(s["date"] for s in sales) + timedelta(days=1)

    # Aggregate data by customer
    agg: Dict[int, Dict[str, Any]] = {}
    for s in sales:
        cid = s["customer_id"]
        if cid not in agg:
            agg[cid] = {"customer_id": cid, "last_date": s["date"], "frequency": 0, "monetary": 0.0}
        agg[cid]["frequency"] += 1
        agg[cid]["monetary"] += float(s["amount"])
        if s["date"] > agg[cid]["last_date"]:
            agg[cid]["last_date"] = s["date"]

    # Build the RFM table
    rfm_rows: List[Dict[str, Any]] = []
    for v in agg.values():
        recency_days = (reference_date - v["last_date"]).days
        rfm_rows.append({
            "customer_id": v["customer_id"],
            "recency_days": recency_days,
            "frequency": v["frequency"],
            "monetary": round(v["monetary"], 2)
        })

    rfm_rows.sort(key=lambda x: x["customer_id"])
    return {
        "reference_date": reference_date.isoformat(),
        "data": rfm_rows,
        "explanation": "This is the RFM table: Recency = days since last purchase, "
                       "Frequency = number of purchases, Monetary = total amount spent. "
                       "It was obtained by aggregating transactions per customer."
    }


@app.get("/rfm/standardized")
def get_rfm_standardized() -> Dict[str, Any]:
    """
    Standardize the RFM metrics using z-scores.

    Returns:
        dict: Contains the reference date and the list of customers with standardized RFM values.
    """
    rfm_response = get_rfm()
    rows = rfm_response["data"]
    if not rows:
        return {"reference_date": rfm_response["reference_date"], "data": [], "explanation": "No RFM data available."}

    # Extract numeric variables
    X = np.array([[r["recency_days"], r["frequency"], r["monetary"]] for r in rows])

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build standardized response
    standardized_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        standardized_rows.append({
            "customer_id": r["customer_id"],
            "recency_z": round(float(X_scaled[i, 0]), 3),
            "frequency_z": round(float(X_scaled[i, 1]), 3),
            "monetary_z": round(float(X_scaled[i, 2]), 3),
        })

    return {
        "reference_date": rfm_response["reference_date"],
        "data": standardized_rows,
        "explanation": "These are the standardized RFM values (z-scores). Each variable was rescaled "
                       "to have mean 0 and standard deviation 1, so they are comparable for clustering."
    }


@app.get("/rfm/clusters")
def get_rfm_clusters() -> Dict:
    """
    Perform automatic KMeans clustering on standardized RFM features.

    Steps:
    1. Retrieve standardized RFM data.
    2. Try different k values (2 to N-1) and evaluate clustering quality using multiple metrics.
    3. Select the best k based on silhouette score.
    4. Merge the best cluster labels back into the original sales transactions.
    
    Returns:
        dict: JSON object containing:
            - reference_date: Date used as reference for recency calculation.
            - best_k: The number of clusters chosen automatically.
            - metrics: List of evaluation metrics for each tested k.
            - data: Original sales transactions enriched with the assigned cluster.
    """
    # 1. Get standardized RFM data
    rfm_std = get_rfm_standardized()
    rows = rfm_std["data"]
    if not rows:
        return {
            "reference_date": rfm_std["reference_date"],
            "best_k": None,
            "metrics": [],
            "data": [],
            "explanation": "No sales data available to perform clustering."
        }

    # Prepare matrix X
    X = np.array([[r["recency_z"], r["frequency_z"], r["monetary_z"]] for r in rows])

    # 2. Try different k values and evaluate
    metrics = []
    best_k = None
    best_score = -1
    best_labels = None

    for k in range(2, min(10, len(rows) - 1) + 1):  # k between 2 and N-1 customers
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        chi = calinski_harabasz_score(X, labels)
        inertia = model.inertia_

        metrics.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": chi,
            "inertia": inertia
        })

        # Keep the best k by silhouette score
        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    # 3. Map customer_id -> cluster
    customer_clusters = {rows[i]["customer_id"]: int(best_labels[i]) for i in range(len(rows))}

    # 4. Enrich the original sales dataset with the cluster of each customer
    clustered_sales = []
    for s in sales:
        clustered_sales.append({
            **s,
            "cluster": customer_clusters.get(s["customer_id"], None)
        })

    return {
        "reference_date": rfm_std["reference_date"],
        "best_k": best_k,
        "metrics": metrics,
        "data": clustered_sales,
        "explanation": (
            "Clusters were computed automatically using standardized RFM values "
            "and assigned back to each transaction in the original sales dataset, "
            "so every sale inherits the cluster of its customer."
        )
    }

@app.get("/rfm/plot")
def plot_rfm_clusters():
    """
    Returns a PNG scatter plot (PCA1 vs PCA2) of RFM with points colored by cluster.
    The human-readable explanation is included in the 'X-Explanation' HTTP header.
    """

    # 1) RFM original
    rfm_response = get_rfm()
    rfm_rows = rfm_response["data"]
    if not rfm_rows:
        return {"message": "No RFM data available"}

    # 2) Clusters
    cluster_response = get_rfm_clusters()
    cluster_rows = cluster_response["data"]
    cluster_map = {row["customer_id"]: row["cluster"] for row in cluster_rows}

    # 3) Merge cluster -> RFM
    data_with_clusters = []
    for row in rfm_rows:
        cid = row["customer_id"]
        if cid in cluster_map:
            row = dict(row)  # evitar mutar la referencia original
            row["cluster"] = cluster_map[cid]
            data_with_clusters.append(row)

    if not data_with_clusters:
        return {"message": "No matching clusters for RFM customers"}

    # 4) Matriz RFM y labels
    X = np.array([[r["recency_days"], r["frequency"], r["monetary"]] for r in data_with_clusters])
    labels = [int(r["cluster"]) for r in data_with_clusters]

    # 5) PCA a 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # 6) Plot
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=80, edgecolor="k")
    for i, r in enumerate(data_with_clusters):
        plt.text(X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02, str(r["customer_id"]), fontsize=8)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Customer Clusters (k={cluster_response['best_k']})")
    plt.colorbar(scatter, label="Cluster")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    # 7) Explicación como header
    explanation = (
    "PCA1 and PCA2 are the first two principal components obtained from Principal Component Analysis (PCA). "
    "They are new synthetic axes that capture most of the variance in the original RFM features "
    "(Recency, Frequency, Monetary). PCA1 explains the largest share of variance across customers, "
    "while PCA2 explains the second largest share, orthogonal to PCA1. "
    "Each point is a customer; its position comes from reducing the 3D RFM space into these 2D coordinates, "
    "and its color represents the assigned cluster. "
    "This plot helps visually inspect how customer segments separate in reduced space."
    )

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Explanation": explanation}
    )
