import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import reader


def plot_family_size_dist(train_targets):
    # Plot the distribution of family sizes
    f, ax = plt.subplots(figsize=(8, 5))
    sorted_targets =  train_targets.groupby(train_targets).size().sort_values(ascending=False)
    sns.histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax)
    plt.title("Distribution of family sizes for the 'train' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")
    plt.savefig('plots/family_size_dist.png')


def plot_sequence_len_dist(train_data):
    # Plot the distribution of sequences' lengths
    f, ax = plt.subplots(figsize=(8, 5))
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()
    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)
    ax.axvline(mean, color='r', linestyle='-', label=f"Mean = {mean:.1f}")
    ax.axvline(median, color='g', linestyle='-', label=f"Median = {median:.1f}")
    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")
    plt.savefig('plots/sequence_len_dist.png')


def get_amino_acid_frequencies(data):
    aa_counter = Counter()
    for sequence in data:
        aa_counter.update(sequence)
    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})


def plot_AA_freq_dist(train_data):
    # Plot the distribution of AA frequencies
    f, ax = plt.subplots(figsize=(8, 5))
    amino_acid_counter = get_amino_acid_frequencies(train_data)
    sns.barplot(x='AA', y='Frequency', data=amino_acid_counter.sort_values(by=['Frequency'], ascending=False), ax=ax)
    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.savefig('plots/AA_freq_dist.png')


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), './random_split')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default=data_dir,
                        help="random_split data path")

    args = parser.parse_args()
    train_data, train_targets = reader("train", args.data_dir)
    plot_family_size_dist(train_targets)
    plot_sequence_len_dist(train_data)
    plot_AA_freq_dist(train_data)