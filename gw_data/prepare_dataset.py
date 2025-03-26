import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SEG_DURATION = 4096
np.random.seed(12)

files = os.listdir("segments")
print(f"# Files: {len(files)}")
num_samples = 0
for file in files:
    interval = file.replace(".npy", '')
    interval = interval.replace("L1-", '')

    duration = int(interval.split('_')[1]) - int(interval.split('_')[0])
    num_samples += duration

print(f"# Samples: {num_samples}")

bbh_indices = np.random.randint(low=0, high=num_samples-1, size=num_samples//4)

allowed_indices = np.setdiff1d(np.arange(num_samples), bbh_indices)
bns_indices = np.random.choice(allowed_indices, size=num_samples//4, replace=False)

categories = np.zeros(num_samples)
categories[bbh_indices] = 1
categories[bns_indices] = 2

m1, m2 = np.zeros(num_samples), np.zeros(num_samples)
snr = np.zeros(num_samples)

m1[bbh_indices] = np.random.uniform(low=20, high=95, size=num_samples//4)
m2[bbh_indices] = np.random.uniform(low=0.1*m1[bbh_indices], high=m1[bbh_indices])

m1[bns_indices] = np.random.uniform(low=1, high=2, size=num_samples//4)
m2[bns_indices] = np.random.uniform(low=1, high=m1[bns_indices])

snr[bbh_indices] = np.random.triangular(left=4.0, mode=5.0, right=20.0, size=num_samples//4)
snr[bns_indices] = np.random.triangular(left=4.0, mode=5.0, right=20.0, size=num_samples//4)

df = pd.DataFrame({
    "index": range(num_samples),
    "category": categories,
    "data_file": ["" for _ in range(num_samples)],
    "m1": m1, "m2": m2, "snr": snr
})

idx = 0
for file in files:
    time_series = np.load(f"segments/{file}")
    for segment in tqdm(range(0, len(time_series)//SEG_DURATION), desc=file):
        np.save(
            f"dataset/gwaves/{file.split('.')[0]}_{segment}.npy",
            time_series[segment*SEG_DURATION: (segment+1)*SEG_DURATION]
        )
        idx += 1

df.to_csv("dataset/gwaves_attr.csv", index=False)

print("Cleaning Up ...")
os.removedirs("segments")
