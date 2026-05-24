import subprocess

CSV_PATH = "merged_datasets/SPLIT2.csv"
OUT_DIR = "./checkpoints"


def main():
    cmd = [
        r"c:\Users\kacpe\source\repos\szum\.venv\Scripts\python.exe",
        "-m",
        "src.train",
        "--csv",
        CSV_PATH,
        "--backbone",
        "dinov2",
        "--temporal",
        "transformer",
        "--output-dir",
        OUT_DIR,
        "--batch-size",
        "8",
        "--epochs",
        "30",
        "--lr",
        "0.0003",
        "--dropout",
        "0.50",
        "--unfreeze-epoch",
        "12",
        "--unfreeze-lr-mult",
        "0.05",
        "--loss",
        "focal",
        "--focal-gamma",
        "2.0",
        "--best-metric",
        "f1",
    ]

    print(f"Komenda: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
