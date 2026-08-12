"""Download the raw dataset into data/raw/.

Usage:  python scripts/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.config import DATA_URL, RAW_CSV

# Fallback mirror (the original R analysis repo by the dataset authors)
MIRROR = ("https://raw.githubusercontent.com/LuisM78/"
          "Appliances-energy-prediction-data/master/energydata_complete.csv")


def main():
    if RAW_CSV.exists():
        print("Already downloaded:", RAW_CSV)
        return
    for url in (DATA_URL, MIRROR):
        try:
            print("Downloading", url)
            urllib.request.urlretrieve(url, RAW_CSV)
            print("Saved to", RAW_CSV)
            return
        except Exception as e:
            print("  failed:", e)
    raise SystemExit("Could not download the dataset from any source.")


if __name__ == "__main__":
    main()
