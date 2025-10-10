from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

'''
Used Car price prediction using linear regression machine learning.
'''

def load_vehicle_data():
    tarball_path = Path("./datasets/vehicles.tgz")
    if not tarball_path.is_file():
        Path("./datasets").mkdir(parents=True, exist_ok=True)
        url = ""
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def main():


if "__name__" == "__main__":
    main()


