import os
import subprocess

# Definiujemy konfigurację dla każdego modelu
models = [
    {
        "name": "MODEL 1 (SPLIT1)",
        "csv": "merged_datasets/SPLIT1.csv",
        "checkpoint": "checkpoints/SPLIT1_dinov2_transformer_20260503_205035/best_model.pt",
        "temporal": "transformer"
    },
    {
        "name": "MODEL 2 (SPLIT2)",
        "csv": "merged_datasets/SPLIT2.csv",
        "checkpoint": "checkpoints/SPLIT2_dinov2_transformer_20260504_005307/best_model.pt",
        "temporal": "transformer"
    },
    {
        "name": "MODEL 3 (SPLIT3)",
        "csv": "merged_datasets/SPLIT3.csv",
        "checkpoint": "checkpoints/SPLIT3_dinov2_transformer_20260503_210606/best_model.pt",
        "temporal": "transformer"
    }
]

# Podzbiory do przetestowania dla każdego modelu
data_splits = ["train", "val", "test"]

def main():
    total_runs = len(models) * len(data_splits)
    current_run = 1
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"ROZPOCZYNAM EWALUACJĘ: {model['name']}")
        print(f"{'='*60}")
        
        for data_split in data_splits:
            print(f"\n---> [{current_run}/{total_runs}] Uruchamianie testu dla podzbioru: {data_split.upper()}...")
            
            cmd = [
                r"c:\Users\kacpe\source\repos\szum\.venv\Scripts\python.exe", "-m", "src.test",
                "--csv", model["csv"],
                "--checkpoint", model["checkpoint"],
                "--backbone", "dinov2",
                "--temporal", model["temporal"],
                "--split", data_split
            ]
            
            print(f"Komenda: {' '.join(cmd)}")
            
            try:
                # Uruchomienie komendy. Oczekujemy na jej zakończenie.
                subprocess.run(cmd, check=True)
                print(f"---> [SUKCES] Zakończono test dla {data_split.upper()}")
            except subprocess.CalledProcessError as e:
                print(f"---> [BŁĄD] Wystąpił błąd podczas ewaluacji podzbioru {data_split.upper()}: {e}")
                
            current_run += 1

    print("\n" + "="*60)
    print("WSZYSTKIE EWALUACJE ZOSTAŁY ZAKOŃCZONE!")
    print("="*60)

if __name__ == "__main__":
    main()
