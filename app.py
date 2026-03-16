import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.title("Movie Review Clustering with PCA")

st.write("Upload a CSV file containing movie reviews.")

uploaded_file = st.file_uploader("Upload reviews.csv", type=["csv"])

if uploaded_file is not None:

    # Load text dataset
    reviews = np.genfromtxt(uploaded_file, delimiter=",", dtype=str, skip_header=1)

    st.subheader("Sample Reviews")
    st.write(reviews[:5])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)

    # Convert to array
    X = X.toarray()

    # PCA Reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X)

    st.subheader("PCA Reduced Data")
    st.write(pca_data[:5])

    # KMeans Clustering
    st.subheader("K-Means Clustering")

    k = st.slider("Number of clusters (K)", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_data)

    sil = silhouette_score(pca_data, labels)
    dbi = davies_bouldin_score(pca_data, labels)

    st.write("Silhouette Score:", sil)
    st.write("Davies-Bouldin Index:", dbi)

    fig, ax = plt.subplots()

    ax.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=labels
    )

    ax.set_title("K-Means Clustering (PCA)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    st.pyplot(fig)

    # DBSCAN Clustering
    st.subheader("DBSCAN Clustering")

    eps = st.slider("EPS", 0.1, 2.0, 0.5)
    min_samples = st.slider("Min Samples", 2, 10, 3)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(pca_data)

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=db_labels
    )

    ax2.set_title("DBSCAN Clustering (PCA)")
    ax2.set_xlabel("PCA1")
    ax2.set_ylabel("PCA2")

    st.pyplot(fig2)

    st.subheader("Cluster Insights")

    st.write(""Cluster examples 
    Cluster 1 → Action / thriller related reviews  
    Cluster 2 → Romantic / emotional movie reviews  
    Cluster 3 → Comedy / entertainment reviews  
    DBSCAN may detect unusual or mixed reviews as noise.""")
