from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from typing import List, Optional, Literal, Dict, Any
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import PCA
from pydantic import BaseModel, Field


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
    Calculate the RFM (recency, Frequency, Monetary) table from the sales dataset.

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
        recency = (reference_date - v["last_date"]).days
        rfm_rows.append({
            "customer_id": v["customer_id"],
            "recency": recency,
            "frequency": v["frequency"],
            "monetary": round(v["monetary"], 2)
        })

    rfm_rows.sort(key=lambda x: x["customer_id"])
    return {
        "reference_date": reference_date.isoformat(),
        "data": rfm_rows,
        "explanation": "This is the RFM table: recency = days since last purchase, "
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
    X = np.array([[r["recency"], r["frequency"], r["monetary"]] for r in rows])

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
    X = np.array([[r["recency"], r["frequency"], r["monetary"]] for r in data_with_clusters])
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
    "(recency, Frequency, Monetary). PCA1 explains the largest share of variance across customers, "
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

# ---------- Pydantic models ----------

class SaleIn(BaseModel):
    sale_id: int
    customer_id: int
    date: date
    amount: float
    product: Optional[str] = None
    channel: Optional[str] = None

class RFMIn(BaseModel):
    customer_id: int
    recency: int = Field(ge=0)
    frequency: int = Field(ge=0)
    monetary: float = Field(ge=0)

class ClusterRequest(BaseModel):
    """
    Payload to run clustering as-a-service.

    You can send either:
      - `sales`: raw transactions (the API will compute RFM), or
      - `rfm`: an already computed RFM table.

    Options:
      - reference_date (optional): if not provided and using sales, it defaults to (max(date)+1 day).
      - k_min / k_max: search range for K.
      - selection_metric: how to pick the best K ('silhouette' | 'calinski_harabasz' | 'davies_bouldin').
      - output_mode: 'per_transaction' | 'per_customer' | 'rfm_only'.
    """
    sales: Optional[List[SaleIn]] = None
    rfm: Optional[List[RFMIn]] = None
    reference_date: Optional[date] = None
    k_min: int = 2
    k_max: int = 10
    selection_metric: Literal["silhouette", "calinski_harabasz", "davies_bouldin"] = "silhouette"
    output_mode: Literal["per_transaction", "per_customer", "rfm_only"] = "per_transaction"

class ClusterResponse(BaseModel):
    reference_date: Optional[str]
    best_k: Optional[int]
    selection_metric: str
    metrics: List[Dict[str, float]]
    cluster_summary: Dict[str, int]
    data: List[Dict[str, Any]]
    explanation: str

# ---------- Helpers ----------

def _compute_rfm_from_sales(sales: List[SaleIn], reference_date: Optional[date]) -> (List[Dict[str, Any]], date):
    """
    Build RFM rows from raw sales.
    Returns (rfm_rows, reference_date).
    """
    if not sales:
        return [], reference_date or date.today()

    if reference_date is None:
        reference_date = max(s.date for s in sales) + timedelta(days=1)

    agg: Dict[int, Dict[str, Any]] = {}
    for s in sales:
        cid = s.customer_id
        if cid not in agg:
            agg[cid] = {"customer_id": cid, "last_date": s.date, "frequency": 0, "monetary": 0.0}
        agg[cid]["frequency"] += 1
        agg[cid]["monetary"] += float(s.amount)
        if s.date > agg[cid]["last_date"]:
            agg[cid]["last_date"] = s.date

    rfm_rows: List[Dict[str, Any]] = []
    for v in agg.values():
        recency = (reference_date - v["last_date"]).days
        rfm_rows.append({
            "customer_id": v["customer_id"],
            "recency": recency,
            "frequency": v["frequency"],
            "monetary": round(v["monetary"], 2),
        })

    rfm_rows.sort(key=lambda x: x["customer_id"])
    return rfm_rows, reference_date

def _pick_best_k(metrics_list: List[Dict[str, float]], selection_metric: str) -> int:
    """
    Select best k according to selection_metric.
    - silhouette: max
    - calinski_harabasz: max
    - davies_bouldin: min
    """
    if not metrics_list:
        return None
    if selection_metric in ("silhouette", "calinski_harabasz"):
        return max(metrics_list, key=lambda m: m[selection_metric])["k"]
    else:  # davies_bouldin
        return min(metrics_list, key=lambda m: m["davies_bouldin"])["k"]

# ---------- New endpoint: clustering-as-a-service ----------

@app.post("/rfm/clusters/apply", response_model=ClusterResponse)
def apply_rfm_clustering(payload: ClusterRequest) -> Dict[str, Any]:
    """
    Apply automatic KMeans clustering given external data.

    Steps:
      1) Use provided RFM rows OR compute RFM from provided sales.
      2) Standardize (z-scores) the RFM features (R,F,M).
      3) Try K in [k_min, ..., k_max] (bounded by N-1) and evaluate metrics.
      4) Select best K according to `selection_metric`.
      5) Return metrics and data labeled according to `output_mode`.

    Returns:
      - reference_date
      - best_k
      - selection_metric
      - metrics (per tested k)
      - cluster_summary (counts per cluster)
      - data (per_transaction | per_customer | rfm_only)
      - explanation
    """
    # Validate input
    if not payload.sales and not payload.rfm:
        raise HTTPException(status_code=400, detail="Provide either `sales` or `rfm` in the payload.")

    # 1) Obtain RFM rows + reference_date
    if payload.rfm:
        rfm_rows = [r.dict() for r in payload.rfm]
        if not rfm_rows:
            return ClusterResponse(
                reference_date=None, best_k=None, selection_metric=payload.selection_metric,
                metrics=[], cluster_summary={}, data=[],
                explanation="Empty RFM payload; nothing to cluster."
            ).dict()
        ref_date = payload.reference_date or date.today()
    else:
        # compute from sales
        rfm_rows, ref_date = _compute_rfm_from_sales(payload.sales, payload.reference_date)
        if not rfm_rows:
            return ClusterResponse(
                reference_date=(ref_date.isoformat() if ref_date else None),
                best_k=None, selection_metric=payload.selection_metric,
                metrics=[], cluster_summary={}, data=[],
                explanation="No sales data available to compute RFM and cluster."
            ).dict()

    # Matrix X (R,F,M)
    X = np.array([[r["recency"], r["frequency"], r["monetary"]] for r in rfm_rows], dtype=float)

    # 2) Standardize
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    n = Xz.shape[0]
    # Boundaries for K
    k_min = max(2, payload.k_min)
    k_max = min(payload.k_max, n - 1) if n > 2 else 1

    if n < 2 or k_max < 2:
        # Not enough customers to form >= 2 clusters
        # Return per requested output mode with cluster=None
        customer_ids = [r["customer_id"] for r in rfm_rows]
        cluster_summary: Dict[str, int] = {}
        data_out: List[Dict[str, Any]] = []

        if payload.output_mode == "per_transaction":
            if not payload.sales:
                explanation = "Not enough customers for clustering; no sales provided to enrich."
                data_out = []
            else:
                sales_out = []
                for s in payload.sales:
                    sales_out.append({**s.dict(), "cluster": None})
                data_out = sales_out
                explanation = "Not enough customers for clustering; returned original transactions with cluster=None."
        elif payload.output_mode == "per_customer":
            data_out = [{"customer_id": cid, "cluster": None} for cid in customer_ids]
            explanation = "Not enough customers for clustering; returned customers with cluster=None."
        else:  # rfm_only
            data_out = [{**r, "cluster": None} for r in rfm_rows]
            explanation = "Not enough customers for clustering; returned RFM with cluster=None."

        return ClusterResponse(
            reference_date=ref_date.isoformat(),
            best_k=None,
            selection_metric=payload.selection_metric,
            metrics=[],
            cluster_summary=cluster_summary,
            data=data_out,
            explanation=explanation
        ).dict()

    # 3) Grid search over K with metrics
    metrics: List[Dict[str, float]] = []
    best_labels = None
    best_k = None

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(Xz)

        sil = silhouette_score(Xz, labels)
        dbi = davies_bouldin_score(Xz, labels)
        chi = calinski_harabasz_score(Xz, labels)
        inertia = model.inertia_

        metrics.append({
            "k": float(k),
            "silhouette": float(sil),
            "davies_bouldin": float(dbi),
            "calinski_harabasz": float(chi),
            "inertia": float(inertia)
        })

    best_k = _pick_best_k(metrics, payload.selection_metric)

    # Refit with best_k to get final labels
    model = KMeans(n_clusters=int(best_k), random_state=42, n_init="auto")
    best_labels = model.fit_predict(Xz).astype(int)

    # Map customer_id -> cluster
    customer_clusters = {rfm_rows[i]["customer_id"]: int(best_labels[i]) for i in range(n)}

    # cluster summary
    cluster_summary: Dict[str, int] = {}
    for c in best_labels:
        key = str(int(c))
        cluster_summary[key] = cluster_summary.get(key, 0) + 1

    # 4) Build output according to output_mode
    data_out: List[Dict[str, Any]] = []
    if payload.output_mode == "per_transaction":
        if not payload.sales:
            # No transactions to enrich; fall back to per_customer
            data_out = [{"customer_id": cid, "cluster": customer_clusters[cid]} for cid in customer_clusters]
            explanation = (
                "Clusters computed on standardized RFM. No transactions were provided; "
                "returning per-customer cluster labels."
            )
        else:
            sales_out = []
            for s in payload.sales:
                sales_out.append({**s.dict(), "cluster": customer_clusters.get(s.customer_id, None)})
            data_out = sales_out
            explanation = (
                "Clusters were computed automatically using standardized RFM values and applied back to each "
                "transaction (per-transaction output)."
            )
    elif payload.output_mode == "per_customer":
        data_out = [{"customer_id": cid, "cluster": customer_clusters[cid]} for cid in customer_clusters]
        explanation = (
            "Clusters computed on standardized RFM. Returning per-customer labels."
        )
    else:  # rfm_only
        data_out = [
            {**rfm_rows[i], "cluster": int(best_labels[i])}
            for i in range(n)
        ]
        explanation = (
            "Clusters computed on standardized RFM. Returning RFM rows labeled with the chosen cluster."
        )

    return ClusterResponse(
        reference_date=ref_date.isoformat(),
        best_k=int(best_k),
        selection_metric=payload.selection_metric,
        metrics=metrics,
        cluster_summary=cluster_summary,
        data=data_out,
        explanation=explanation + " Selection metric: {}.".format(payload.selection_metric)
    ).dict()
