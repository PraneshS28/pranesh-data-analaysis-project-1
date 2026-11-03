import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page title
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.write("K-Means Clustering using Mall Customers Dataset")

# Load data
df = pd.read_csv("Customer_Segments.csv")

# Sidebar Filters
st.sidebar.header("Filter Customers")

# Gender filter
gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.selectbox(
    "Select Gender",
    gender_options
)

# Age filter
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

selected_age = st.sidebar.slider(
    "Select Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Apply filters
filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

filtered_df = filtered_df[
    (filtered_df["Age"] >= selected_age[0]) &
    (filtered_df["Age"] <= selected_age[1])
]



# KPI Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(filtered_df))

with col2:
    st.metric(
        "Average Income",
        f"{filtered_df['Annual Income (k$)'].mean():.1f} k$"
    )

with col3:
    st.metric(
        "Average Spending Score",
        f"{filtered_df['Spending Score (1-100)'].mean():.1f}"
    )


st.divider()

# Customer Segments Chart
st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    filtered_df["Annual Income (k$)"],
    filtered_df["Spending Score (1-100)"],
    c=filtered_df["Cluster"],
    cmap="rainbow",
    s=70
)

# Centroids
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)

ax.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    c="black",
    s=250,
    marker="X",
    label="Centroids"
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

st.divider()

# Cluster Distribution
st.subheader("Customers in Each Cluster")

cluster_count = filtered_df["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_count)

st.divider()

# Show Dataset
st.subheader("Segmented Customer Data")

st.dataframe(filtered_df)
st.divider()

# Business Insights
st.subheader("Business Insights")

st.markdown("""
- **Cluster 0 – Average Customers:** Offer loyalty rewards and regular promotions.
- **Cluster 1 – High-Value Customers:** Provide VIP memberships and exclusive offers.
- **Cluster 2 – Low-Income, High-Spending:** Offer affordable products and attractive discounts.
- **Cluster 3 – High-Income, Low-Spending:** Use personalized promotions to encourage purchases.
- **Cluster 4 – Low-Income, Low-Spending:** Promote budget-friendly products and seasonal discounts.
""")
st.divider()

# Project Conclusion
st.subheader("Project Conclusion")

st.write("""
The Customer Segmentation Project successfully grouped 200 mall customers
into five distinct segments using the K-Means clustering algorithm.

The analysis used annual income and spending score to identify different
customer behaviors.

These insights can help businesses create targeted marketing strategies,
improve customer satisfaction, offer personalized promotions, and increase
customer loyalty.

Overall, this project demonstrates how machine learning and data analytics
can support better business decisions.
""")