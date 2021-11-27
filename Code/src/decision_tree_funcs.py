import numpy as np
import pandas as pd


def get_all_potential_splits(vectors_df):
    all_potential_col_splits = {}
    for col in vectors_df.columns[:-1]:
        all_potential_col_splits[col] = []
        uniques = vectors_df[col].unique()
        uniques.sort()

        for i in range(1, len(uniques)):
            all_potential_col_splits[col] = all_potential_col_splits[col] + [uniques[i] + uniques[i - 1] / 2]

    return all_potential_col_splits


def split_col_data(vectors_df, col, split_value):
    data_below = vectors_df[vectors_df[col] <= split_value]
    data_above = vectors_df[vectors_df[col] > split_value]

    return data_below, data_above


def calculate_entropy(data):
    label_column = data.iloc[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(vectors_df, potential_splits):
    # Start with a high entropy value
    overall_entropy = 9999
    for col in potential_splits.keys():
        for value in potential_splits[col]:
            data_below, data_above = split_col_data(vectors_df, col, value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = col
                best_split_value = value

    return best_split_column, best_split_value