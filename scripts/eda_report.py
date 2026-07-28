from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "diabetes+130-us+hospitals+for+years+1999-2008" / "diabetic_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_eda():
    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", np.nan)
    df["readmitted_binary"] = df["readmitted"].apply(lambda value: 1 if str(value).strip() == "<30" else 0)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="readmitted_binary")
    plt.title("Readmission Class Distribution")
    plt.xlabel("Readmitted within 30 days")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["time_in_hospital"].astype(float), kde=True)
    plt.title("Distribution of Length of Stay")
    plt.xlabel("Time in Hospital")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "length_of_stay_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="readmitted_binary", y="num_medications")
    plt.title("Medication Count by Readmission Status")
    plt.xlabel("Readmitted within 30 days")
    plt.ylabel("Number of Medications")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "medication_count_by_readmission.png")
    plt.close()

    print("EDA plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    generate_eda()
