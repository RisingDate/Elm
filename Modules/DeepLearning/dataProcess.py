import pandas as pd


def data_process(path):
    data = pd.read_csv(path, sep="\t")

    return data
