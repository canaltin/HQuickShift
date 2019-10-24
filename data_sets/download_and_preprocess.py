import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import subprocess
import requests


def setup_directory(dir_name, verbose=False):
    """
    Setup directory in case it does not exist
    Parameters:
    -------------
    dir_name: str, path + name to directory
    verbose: bool, indicates whether directory creation should be printed or not.
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if verbose:
                print("Created Directory: {}".format(dir_name))
        except Exception as e:
            print("Could not create directory: {}\n {}".format(dir_name, e))


def unzip_file(url)->pd.DataFrame:
    r = requests.get(url, allow_redirects=True)
    fname = url.split(r"/")[-1]
    print(fname)
    with open(fname, "wb") as f:
        f.write(r.content)

    fname_cleaned = fname.split(".")[0] + ".data"
    subprocess.run(["uncompress", fname_cleaned])
    arr = np.loadtxt(fname_cleaned)
    df = pd.DataFrame(arr)
    os.remove(fname_cleaned)
    return df


def download(url)->pd.DataFrame:
    if "page-blocks" in url:
        df = unzip_file(url)
    elif "seeds_dataset" in url:
        arr = np.loadtxt(url)
        df = pd.DataFrame(arr)
    elif "phoneme" in url:
        df = pd.read_csv(url, skiprows=[0, 1], header=None)
        df = df.drop([0, 258], 1)
    else:
        df = pd.read_csv(url, header=None)
    if "letter-recognition" in url:
        df = df.drop(0, 1)
    df.columns = [i for i in range(df.shape[1] - 1)] + ["label"]
    return df


def convert_labels(labels: list)->list:
    unique_labels = list(set(labels))
    label_dic = {}
    for i, l_i in enumerate(unique_labels):
        label_dic[l_i] = i

    return [label_dic[l_i] for l_i in labels]


def preprocess(df)->pd.DataFrame:
    scaled = scale(df.drop("label", 1))
    scaled_df = pd.DataFrame(scaled)
    converted_labels = convert_labels(df["label"].tolist())
    scaled_df["label"] = converted_labels
    return scaled_df


def get_uci_urls()->list:
    urls = [
        "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/phoneme.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.data.Z",
    ]
    return urls


def save(url, df, data_dir=os.path.join(os.pardir, "data")):
    data_set_name = os.path.split(url)[-1]
    data_set_name = data_set_name.split(".")[0]
    dir_path = os.path.join(data_dir, data_set_name)
    setup_directory(dir_path)
    file_path = os.path.join(dir_path, data_set_name + ".csv")
    df.to_csv(file_path, header=None, index=False)


def main():
    urls = get_uci_urls()
    for url in urls:
        print("\n#################\nURL: ", url)
        df = download(url)
        print("Before Preprocessing")
        print(df.head())
        df = preprocess(df)
        print("After Preprocessing")
        print(df.head())
        save(url, df)


if __name__ == '__main__':
    main()
