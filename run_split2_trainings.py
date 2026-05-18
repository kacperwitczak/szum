import os
import subprocess

# Ten skrypt uruchomi 6 eksperymentów na SPLIT2 (3 architektury temporalne x 2 architektury wizyjne)
# Wszystkie ze stale zamrożonym backbonem (zgodnie z życzeniem)

CSV_PATH = "merged_datasets/SPLIT2.csv"
OUT_DIR = "./checkpoints"

temporals = ["lstm", "gru", "transformer"]
backbones = ["dinov2", "efficientnet"]

def main():
    total_runs = len(temporals) * len(backbones)
    current_run = 1
    
    for backbone in backbones:
        for temporal in temporals:
            # Pomiń zakończone już DINOv2
            if backbone == "dinov2":
                continue
                
            print(f"\n{'='*60}")
            print(f"ROZPOCZYNAM TRENING [{current_run}/{total_runs}]: SPLIT2 | {backbone.upper()} + {temporal.upper()}")
            print(f"{'='*60}\n")
            
            cmd = [
                r"c:\Users\kacpe\source\repos\szum\.venv\Scripts\python.exe", "-m", "src.train",
                "--csv", CSV_PATH,
                "--backbone", backbone,
                "--temporal", temporal,
                "--output-dir", OUT_DIR,
                "--batch-size", "8", # możesz łatwo dostosować te parametry
                "--epochs", "30",
                "--loss", "focal",
                "--focal-gamma", "2.0"
                # parametry typu --unfreeze-epoch pominięte -> backbone zostaje zamrożony przez cały trening
            ]
            
            print(f"Komenda: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, check=True)
                print(f"\n---> [SUKCES] Zakończono trening dla {backbone.upper()} + {temporal.upper()}")
            except subprocess.CalledProcessError as e:
                print(f"\n---> [BŁĄD] Trening przerwany z błędem dla {backbone.upper()} + {temporal.upper()}: {e}")
                
            current_run += 1

    print("\n" + "="*60)
    print("WSZYSTKIE TRENINGI NA SPLIT2 ZOSTAŁY ZAKOŃCZONE!")
    print("="*60)

if __name__ == "__main__":
    main()
