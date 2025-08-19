# Customer Segmentation API (RFM + Clustering)

This API provides tools for performing **RFM (Recency, Frequency, Monetary) analysis** and customer segmentation using clustering techniques.  
It is built with **FastAPI** and includes endpoints to compute RFM values, assign clusters, and visualize customer segmentation in 2D space using PCA.  

---

## 🚀 Features
- **RFM Analysis**: Calculates Recency (days since last purchase), Frequency (number of purchases), and Monetary (total amount spent) per customer.  
- **Clustering**: Applies machine learning (K-Means) to segment customers into groups based on RFM behavior.  
- **Visualization**: Provides a 2D scatter plot using PCA (Principal Component Analysis) to project RFM values into two dimensions, making clusters easier to interpret.  
- **Explanations**: Each endpoint includes human-readable explanations of the process and results.  

---

## 📦 Installation

Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Run the API locally with Uvicorn:
uvicorn app:app --reload

The API will be available at:
👉 http://127.0.0.1:8000

## 📊 Endpoints

### `GET /rfm`
- Returns the RFM table for all customers.  
- Explanation included in the response: what RFM is and how values were computed.  

### `GET /rfm/standardized`
- Returns the **standardized RFM table** using z-scores: `recency_z`, `frequency_z`, `monetary_z` (via `StandardScaler`).  
- **Explanation** included: why standardization is needed (to put features on comparable scale) and that this output is the **input for clustering**.

### `GET /rfm/clusters`
- Returns the RFM table with cluster assignments.  
- Explains how K-Means clustering was applied and what each cluster represents.  

### `GET /rfm/plot`
- Returns a PNG image of customers plotted in 2D using PCA.  
- **PCA1** = first principal component (captures most variance in RFM).  
- **PCA2** = second principal component (captures second-most variance, orthogonal to PCA1).  
- Each point = customer, **color = cluster**.  
- Response also includes an explanation of what PCA1 and PCA2 mean and what the visualization shows.  



### `POST /rfm/clusters/apply`

This endpoint applies the RFM segmentation with automatic clustering on your customer data.

🔹 Request

Method: POST
Path:

POST /rfm/clusters/apply
Content-Type: application/json


You can send the data in two different ways:

1. With raw sales data (sales)

The backend will automatically calculate the Recency, Frequency, Monetary (RFM) metrics from the transactions.

Request example:

{
  "sales": [
    {
      "sale_id": 1,
      "customer_id": 101,
      "date": "2025-01-05",
      "amount": 120,
      "product": "Laptop",
      "channel": "Online"
    },
    {
      "sale_id": 2,
      "customer_id": 101,
      "date": "2025-02-10",
      "amount": 80,
      "product": "Mouse",
      "channel": "Store"
    }
  ],
  "selection_metric": "silhouette", 
  "output_mode": "per_transaction"
}


selection_metric: metric used to choose the best number of clusters (silhouette, davies_bouldin, calinski_harabasz).

output_mode: can be "per_transaction" (cluster assigned to each transaction) or "per_customer" (one cluster per customer).

2. With precomputed RFM data (rfm)

If you already calculated the metrics, you can provide them directly.

Request example:

{
  "rfm": [
    { "customer_id": 101, "recency_days": 20, "frequency": 5, "monetary": 500.0 },
    { "customer_id": 102, "recency_days": 10, "frequency": 3, "monetary": 350.0 },
    { "customer_id": 103, "recency_days": 30, "frequency": 2, "monetary": 120.0 },
    { "customer_id": 104, "recency_days": 5,  "frequency": 8, "monetary": 1000.0 }
  ],
  "selection_metric": "silhouette",
  "output_mode": "per_customer"
}

🔹 Explanation

The service tests multiple values of k (number of clusters).

It evaluates them using the chosen metric (silhouette, davies_bouldin, or calinski_harabasz).

Returns the best k and assigns each customer/transaction to a cluster.

Provides a summary with the number of customers in each cluster (cluster_summary).
---

## 📖 How It Works
1. **Data Input**: The API assumes access to customer transaction data.  
2. **RFM Calculation**: Customers are scored on Recency, Frequency, and Monetary dimensions.  
3. **Clustering**: Customers are grouped into clusters (K chosen automatically).  
4. **Visualization**: PCA reduces 3D RFM to 2D for plotting.  

---

## 🛠 Tech Stack
- [FastAPI](https://fastapi.tiangolo.com/) – Web framework  
- [scikit-learn](https://scikit-learn.org/) – Machine learning (PCA, KMeans)  
- [matplotlib](https://matplotlib.org/) – Data visualization  
- [pandas](https://pandas.pydata.org/) – Data manipulation  

---

## 📌 Example Use Case
- A marketing team wants to **segment customers** into groups (loyal, occasional, at risk, inactive).  
- The API provides **RFM scores + clusters** and a **visual plot** to better understand how customers are distributed.  
- Segments can then be used for **targeted marketing campaigns**.  
