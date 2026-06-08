import csv
from pathlib import Path
import matplotlib.pyplot as plt

base = Path(__file__).parent

def read_scores(filename):
    path = base / filename
    scores = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        score_col = None
        for name in fieldnames:
            if name.lower() == "score":
                score_col = name
                break

        if score_col is None:
            score_col = fieldnames[-1]

        for row in reader:
            scores.append(float(row[score_col]))

    return scores

for i in range(5):
    map_name = f"map{i}"

    aco_scores = read_scores(f"{map_name}_aco_20_tests.csv")
    ga_scores = read_scores(f"{map_name}_ga-aco_20_tests.csv")

    plt.figure(figsize=(6, 5))

    plt.boxplot(
        [aco_scores, ga_scores],
        labels=["ACO", "GA-ACO"],
        showmeans=True
    )

    plt.ylabel("Score")
    plt.title(f"Map{i} 20-run Score Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = base / f"{map_name}_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)

print("All boxplots generated.")