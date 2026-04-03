# SZUM - Python Environment Setup

Ten projekt zawiera skrypty i notebooki do analizy danych WLASL oraz przetwarzania wideo.

## Wymagania

- Python 3.13
- Windows PowerShell
- WLASL dataset -- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?resource=download

## Szybki start (Windows)

1. Przejdz do katalogu projektu:

   ```powershell
   cd d:\college\sem_mag_1\szum
   ```

2. Utworz i aktywuj wirtualne srodowisko:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Pobierz dataset i umieść go folderze kaggle_datset/videos 

4. Zaktualizuj pip i zainstaluj zaleznosci:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Dodaj kernel do Jupyter:

   ```powershell
   python -m ipykernel install --user --name szum-venv --display-name "Python (szum-venv)"
   ```

6. Uruchom Jupyter:

   ```powershell
   jupyter lab
   ```

   lub:

   ```powershell
   jupyter notebook
   ```

## Praca z notebookami

Notebooki sa w katalogu `notebooks/`.
Po uruchomieniu Jupyter wybierz kernel `Python (szum-venv)`.

## MS-ASL: Pobieranie i wycinanie (pilot)

Do pobierania klipow MS-ASL z URL-i uzywamy narzedzi `yt-dlp` i `ffmpeg`.

Wymagania dodatkowe:

- `yt-dlp` (instalowany z `requirements.txt`),
- `ffmpeg` dostepny w systemowym `PATH`. -- `winget install ffmpeg`

Pilot jest przygotowany dla 100 rekordow:

- pierwsze 50 rekordow: wycinanie po klatkach,
- drugie 50 rekordow: wycinanie po czasie (`start_time` / `end_time`).

Pliki wyjsciowe klipow zapisywane sa do katalogu `videos_MS_ASL/`,
a statusy i metadane procesu do CSV manifestu.

### Research Use Notice

Pobieranie materialow wideo jest wykonywane wylacznie do celow badawczych/edukacyjnych.
Uzytkownik odpowiada za zgodnosc z warunkami platformy zrodlowej (np. YouTube ToS),
licencjami danych i obowiazujacym prawem.

## Rozwiazywanie problemow

- Jezeli `py` nie dziala, sprawdz komende `python --version`.
- Jezeli PowerShell blokuje aktywacje:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

- Przy aktualizacji zaleznosci:

  ```powershell
  pip install --upgrade -r requirements.txt
  ```
