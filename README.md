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
