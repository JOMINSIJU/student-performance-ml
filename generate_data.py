import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 500

study_hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)
previous_score = rng.uniform(20, 100, n)

noise = rng.normal(0, 5, n)

final_score = (
    5
    + 3 * study_hours
    + 0.15 * attendance
    + 0.45 * previous_score
    + noise
)

final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours.round(2),
    "attendance": attendance.round(2),
    "previous_score": previous_score.round(2),
    "final_score": final_score.round(2)
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)

print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Dataset shape:", df.shape)
