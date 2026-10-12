import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set to a CSV file path to use your own customer data; keep None for demo data.
CSV_PATH = None
OUTPUT_DIR = "customer_segmentation_output"
RANDOM_STATE = 42
N_CLUSTERS = 5


def make_sample_data(n=300, random_state=42):
    """Create realistic demo customer data when no CSV is provided."""
    rng = np.random.default_rng(random_state)
    profiles = [
        # age center, income center, spending center, profile size
        (24, 32, 78, 65),
        (47, 28, 22, 55),
        (32, 78, 80, 60),
        (42, 82, 25, 55),
        (29, 55, 50, 65),
    ]
    rows = []
    for age_c, income_c, spend_c, size in profiles:
        for _ in range(size):
            rows.append({
                "Age": int(np.clip(rng.normal(age_c, 6), 18, 75)),
                "Annual Income (k$)": round(float(np.clip(rng.normal(income_c, 12), 10, 150)), 1),
                "Spending Score (1-100)": int(np.clip(rng.normal(spend_c, 10), 1, 100)),
                "Purchase Frequency": int(np.clip(rng.normal(8 + spend_c / 15, 3), 1, 30)),
            })
    data = pd.DataFrame(rows)
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_data():
    if CSV_PATH and os.path.exists(CSV_PATH):
        data = pd.read_csv(CSV_PATH)
        # Normalize common column names to the names used by this project.
        aliases = {
            "age": "Age",
            "annual income": "Annual Income (k$)",
            "annual income (k$)": "Annual Income (k$)",
            "income": "Annual Income (k$)",
            "spending score": "Spending Score (1-100)",
            "spending score (1-100)": "Spending Score (1-100)",
            "spending_score": "Spending Score (1-100)",
            "purchase frequency": "Purchase Frequency",
            "purchase_frequency": "Purchase Frequency",
        }
        rename = {col: aliases[col.strip().lower()] for col in data.columns
                  if col.strip().lower() in aliases}
        data = data.rename(columns=rename)
        required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(
                f"Your CSV is missing these columns: {missing}. "
                "Include Age, Annual Income (k$), and Spending Score (1-100)."
            )
        if "Purchase Frequency" not in data.columns:
            data["Purchase Frequency"] = np.nan
        return data
    return make_sample_data()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    data = data.copy()
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=features).reset_index(drop=True)

    if len(data) < N_CLUSTERS:
        raise ValueError(f"Please provide at least {N_CLUSTERS} valid customer rows.")

    # Scale features so age, income, and spending score contribute comparably.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    data["Cluster"] = model.fit_predict(scaled) + 1

    # Give each cluster a readable description based on average spending/income.
    profile = data.groupby("Cluster").agg(
        Customers=("Cluster", "size"),
        Average_Age=("Age", "mean"),
        Average_Annual_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean"),
    ).round(1)

    def describe(row):
        income = row["Average_Annual_Income"]
        spending = row["Average_Spending_Score"]
        if income >= data["Annual Income (k$)"].median() and spending >= 60:
            return "High-value / big spenders"
        if income >= data["Annual Income (k$)"].median() and spending < 40:
            return "Affluent but careful spenders"
        if income < data["Annual Income (k$)"].median() and spending >= 60:
            return "Enthusiastic spenders"
        if spending < 40:
            return "Low-spending customers"
        return "Average / balanced customers"

    profile["Segment Description"] = profile.apply(describe, axis=1)
    data = data.merge(profile["Segment Description"], left_on="Cluster", right_index=True)

    # Save the outputs.
    output_csv = os.path.join(OUTPUT_DIR, "segmented_customers.csv")
    profile_csv = os.path.join(OUTPUT_DIR, "segment_summary.csv")
    data.to_csv(output_csv, index=False)
    profile.to_csv(profile_csv)

    # 1. Scatter plot: annual income vs spending score.
    plt.figure(figsize=(10, 6))
    for cluster in sorted(data["Cluster"].unique()):
        part = data[data["Cluster"] == cluster]
        plt.scatter(part["Annual Income (k$)"], part["Spending Score (1-100)"],
                    label=f"Cluster {cluster}", alpha=0.75, s=42)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments: Income vs Spending Score")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "customer_segments.png"), dpi=160)
    plt.show()

    # 2. Bar chart: number of customers in each cluster.
    plt.figure(figsize=(8, 5))
    counts = data["Cluster"].value_counts().sort_index()
    plt.bar([f"Cluster {i}" for i in counts.index], counts.values)
    plt.xlabel("Customer Cluster")
    plt.ylabel("Number of Customers")
    plt.title("Number of Customers in Each Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_sizes.png"), dpi=160)
    plt.show()

    # 3. Average profile chart.
    profile[["Average_Age", "Average_Annual_Income", "Average_Spending_Score"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Average Customer Characteristics by Segment")
    plt.xlabel("Customer Cluster")
    plt.ylabel("Average value (original units)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "segment_profiles.png"), dpi=160)
    plt.show()

    print("\n========== CUSTOMER SEGMENTATION SUMMARY ==========")
    print(profile.to_string())
    print(f"\nNumber of customers analyzed: {len(data)}")
    print(f"Number of clusters: {N_CLUSTERS}")
    if len(data) > N_CLUSTERS:
        print(f"Silhouette score (higher is generally better): "
              f"{silhouette_score(scaled, model.labels_):.3f}")
    print("\nFiles saved in:", os.path.abspath(OUTPUT_DIR))
    print(" - segmented_customers.csv")
    print(" - segment_summary.csv")
    print(" - customer_segments.png")
    print(" - cluster_sizes.png")
    print(" - segment_profiles.png")


if _name_ == "_main_":
    main()
