import pandas as pd
import numpy as np
from pathlib import Path

base_csv_path = Path("merged_datasets/split8c_frames_signer_disjoint.csv")
df = pd.read_csv(base_csv_path)

df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

print(f"Baza: TRAIN={len(df_train)}, VAL={len(df_val)}, TEST={len(df_test)}")

df_split2 = df.copy()
split2_out = Path("merged_datasets/SPLIT2.csv")
df_split2.to_csv(split2_out, index=False)
print(f"Zapisano {split2_out} (TRAIN={len(df_split2[df_split2['split']=='train'])}, VAL={len(df_split2[df_split2['split']=='val'])})")

df_train_fraction = df_train.sample(frac=0.10, random_state=42)

df_split1 = pd.concat([df_train_fraction, df_val, df_test]).reset_index(drop=True)
split1_out = Path("merged_datasets/SPLIT1.csv")
df_split1.to_csv(split1_out, index=False)
print(f"Zapisano {split1_out} (TRAIN={len(df_train_fraction)}, VAL={len(df_val)})")

df_split3_combined_train = pd.concat([df_train, df_val]).reset_index(drop=True)
df_split3_combined_train["split"] = "train"

leak_val = df_split3_combined_train.sample(frac=0.20, random_state=42).copy()
leak_val["split"] = "val"

df_split3 = pd.concat([df_split3_combined_train, leak_val, df_test]).reset_index(drop=True)
split3_out = Path("merged_datasets/SPLIT3.csv")
df_split3.to_csv(split3_out, index=False)
print(f"Zapisano {split3_out} (TRAIN={len(df_split3_combined_train)}, VAL={len(leak_val)})")
