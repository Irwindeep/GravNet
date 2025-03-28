import os
import pandas as pd
import numpy as np
from gravnet.utils.gw_injection import (
    generate_simulated_waveform,
    adjust_simulated_waveform
)
from tqdm.auto import tqdm

os.environ["LAL_DATA_PATH"] = "gw_data/lalsuite"
df = pd.read_csv("gw_data/dataset/gwaves_attr.csv")
df = df[df["category"] > 0]

total = len(df)
pbar = tqdm(total=total, desc="Generating Simulated Waveforms")
for idx, row in df.iterrows():
    approximant = "SEOBNRv4_ROM" if row["category"] == 1 else "TaylorF2"
    m1, m2 = row["m1"], row["m2"]

    waveform = generate_simulated_waveform(approximant, m1, m2, 20, 1/4096)
    waveform = adjust_simulated_waveform(waveform, desired_length=1)

    np.save(f"gw_data/dataset/simulations/{row['data_file']}", waveform.data)
    pbar.update(1)

pbar.close()
