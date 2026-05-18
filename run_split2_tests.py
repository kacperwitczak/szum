import os
import subprocess

checkpoints_dir = "checkpoints"
csv_file = "merged_datasets/SPLIT2.csv"
data_splits = ["test"]
python_exe = r"c:\Users\kacpe\source\repos\szum\.venv\Scripts\python.exe"

def main():
    # Pobieramy wszystkie foldery w checkpoints zaczynające się od 'SPLIT2_'
    folders = [f for f in os.listdir(checkpoints_dir) if f.startswith("SPLIT2_") and os.path.isdir(os.path.join(checkpoints_dir, f))]

    models = []
    for folder in folders:
        parts = folder.split('_')
        # Struktura nazwy to zazwyczaj: SPLIT2_{backbone}_{temporal}_{data}_{czas}
        if len(parts) >= 3:
            backbone = parts[1]
            temporal = parts[2]
            checkpoint_path = os.path.join(checkpoints_dir, folder, "best_model.pt")
            
            if os.path.exists(checkpoint_path):
                models.append({
                    "name": folder,
                    "csv": csv_file,
                    "checkpoint": checkpoint_path,
                    "backbone": backbone,
                    "temporal": temporal
                })
            else:
                print(f"[Ostrzeżenie] Brak 'best_model.pt' w folderze {folder}. Pomijam.")
        else:
            print(f"[Ostrzeżenie] Folder {folder} nie pasuje do schematu SPLIT2_backbone_temporal_... Pomijam.")

    if not models:
        print("Nie znaleziono żadnych pełnych modeli SPLIT2 z 'best_model.pt'.")
        return

    total_runs = len(models) * len(data_splits)
    current_run = 1
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"ROZPOCZYNAM EWALUACJĘ: {model['name']}")
        print(f"{'='*60}")
        
        for data_split in data_splits:
            print(f"\n---> [{current_run}/{total_runs}] Uruchamianie testu dla podzbioru: {data_split.upper()}...")
            
            cmd = [
                python_exe, "-m", "src.test",
                "--csv", model["csv"],
                "--checkpoint", model["checkpoint"],
                "--backbone", model["backbone"],
                "--temporal", model["temporal"],
                "--split", data_split
            ]
            
            print(f"Komenda: {' '.join(cmd)}")
            
            try:
                # Uruchomienie komendy ewaluacyjnej
                subprocess.run(cmd, check=True)
                print(f"---> [SUKCES] Zakończono test dla {data_split.upper()}")
            except subprocess.CalledProcessError as e:
                print(f"---> [BŁĄD] Wystąpił błąd podczas ewaluacji podzbioru {data_split.upper()}: {e}")
                
            current_run += 1

    print("\n" + "="*60)
    print("WSZYSTKIE EWALUACJE SPLIT2 ZOSTAŁY ZAKOŃCZONE!")
    print("="*60)

if __name__ == "__main__":
    main()