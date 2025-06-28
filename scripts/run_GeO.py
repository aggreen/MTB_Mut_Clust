### script to execute GeO score calculation for a single protein
### meant to be executed as a standalone job

import os
import warnings
import glob
import sys
import pandas as pd
import numpy as np
from evcouplings.compare import DistanceMap
from copy import deepcopy
from scipy.stats import ks_2samp
from evcouplings.visualize.pymol import (
    pymol_pair_lines, pymol_mapping
)
import time

import warnings
warnings.filterwarnings("ignore")

def compute_weight_matrix(dist_matrix):
    """
    Creates a weight matrix using the input distance matrix
    
    Parameters
    ----------
    dist_matrix: Evcouplings.compare.DistanceMap.dist_matrix
    
    Returns
    -------
    np.array
        LxL matrix of 1/distance for all pairs of residues, 0 on diagonal
    """
    
    weights = np.power(dist_matrix, -1)
    np.fill_diagonal(weights, 0)

    return weights

def create_X(feature_df, positions_in_distmap, column="total_mutations"):
    """
    Parameters:
    feature_df: pd.DataFrame
     with columns aa for amino acid position and total_mutations for number of mutations in that site
    
    positions_in_distmap: pd.Series
        list of L positions in the distance map
        
    column: str, optional (default="total_mutations")
        override use of total_mutations column for different string
        
    Returns:
    np.array: an Lx1 matrix counting the number of mutations in each position of the distance map
    
    """
    # initialize X with all zeros
    X = np.zeros(shape=(len(positions_in_distmap.id), 1), dtype=float)
    
    # Get the index of each position
    # note that pandas 
    id_to_index = {int(x):int(idx) for idx,x in enumerate(positions_in_distmap.id)}
    
    # For positions with mutations, replace the value in the X matrix
    for _, row in feature_df.iterrows():
        
        x_index = id_to_index[row.residue.astype(int)]
        X[x_index,0] = float(row[column])
        
    return X

def calculate_G_scores(X, w):
    """
    Calculates the Getis-Ord statistic for all L positions in the structure
    
    Parameters:
    X: np.array
        an Lx1 matrix of the mutations for each
    
    w: np.array
        an LxL weight matrix 
    
    """
    G_scores = np.zeros(X.shape)
    
    # calculate number of positions in the DistanceMap
    n = w.shape[0]

    # Iterate through each position in the distance map
    for i in range(X.shape[0]):
        
        # Need to calculate the GeO score WITHOUT using the number of mutations
        # at the current position
        zeroed_X = deepcopy(X)
        zeroed_X[i] = 0
        
        # sum of weights times X. Recall w[i,i] is 0
        sum_of_wx = np.dot(w[i,:].reshape(1,-1), X)[0][0]

        # mean of x times sum of weights
        mean_x_times_weights = np.mean(zeroed_X) * np.sum(w[i, :])

        S = np.sqrt(np.sum(np.power(zeroed_X,2))/n - np.power(np.mean(zeroed_X),2))

        K = np.sqrt((n * np.sum(np.power(w[i,:], 2)) - np.power(np.sum(w[i,:]),2))/(n-1))

        # Compute G_score
        G_i = (sum_of_wx - mean_x_times_weights) / (S * K)  

        G_scores[i,0] = G_i
        
    return G_scores

def random_G_score_table(runs, X, w, dm):
    # Simulating random distributions of mutations

    column_dict = {}
    column_dict["i"] = list(dm.residues_i.id)
    random_X = deepcopy(X)
    for r in range(runs):
        np.random.shuffle(random_X)

        G_scores = calculate_G_scores(random_X, w)
        column_dict[f"iteration_{r}"] = G_scores.flatten()
    shuffle_table = pd.DataFrame(column_dict)
    return shuffle_table


_, uid, rv, entry, NUM_SHUFFLES, output_path, absolute_path = sys.argv

NUM_SHUFFLES = int(NUM_SHUFFLES)

# read the distance map
dm = DistanceMap.from_file(f"{absolute_path}/filtered_distmaps/{entry}")

# read the mutations to analyze
try:
	to_analyze = pd.read_csv(f"{absolute_path}/{output_path}/{uid}/all_mutations_analyzed.csv", index_col=0)
except:
	to_analyze = pd.read_csv(f"{absolute_path}/{output_path}/{uid}/mutations_analyzed.csv", index_col=0)

# Compute G scores
w = compute_weight_matrix(dm.dist_matrix)
X = create_X(to_analyze, dm.residues_i, column="summed")
G_scores = calculate_G_scores(X, w)
df = pd.DataFrame([dm.residues_i.id.values,G_scores.flatten()]).T
df.columns = ["residue", "G_score"]
df.to_csv(f"{absolute_path}/{output_path}/{uid}/G_scores.csv")
G_score_df = df

# Make the random permutations
shuffle_table = random_G_score_table(NUM_SHUFFLES, X, w, dm)
shuffle_table.to_csv(f"{absolute_path}/{output_path}/{uid}/random_GeO_iterations_{NUM_SHUFFLES}.csv.gz", compression='gzip')

# Computing results from the shuffle table
max_GeO_greater = np.sum(G_score_df.G_score.max() > shuffle_table.max(axis=0)[1::])
mean_GeO_greater = np.sum(G_score_df.G_score.mean() > shuffle_table.mean(axis=0)[1::])
min_GeO_greater = np.sum(G_score_df.G_score.min() > shuffle_table.min(axis=0)[1::])

ks_results = []

for i in range(1,NUM_SHUFFLES+1):
    ks = ks_2samp(G_scores.flatten(), shuffle_table.iloc[:,i].values.flatten())
    ks_results.append([ks.statistic, ks.pvalue])

result_table = pd.DataFrame(ks_results, columns=["score", "pvalue"])
result_table.sort_values("pvalue", inplace=True)
result_table.to_csv(f"{absolute_path}/{output_path}/{uid}/random_GeO_pvalues_{NUM_SHUFFLES}.csv.gz", compression='gzip')

### Now compute GeO versus combination of all shuffled isolates
ks = ks_2samp(G_scores.flatten(), shuffle_table.iloc[:,1::].values.flatten())
result_table_full = pd.DataFrame([[ks.statistic, ks.pvalue, max_GeO_greater, mean_GeO_greater, min_GeO_greater]],
								 columns=["score", "pvalue", "max_GeO_greater", "mean_GeO_greater", "min_GeO_greater"]
								)
result_table_full.to_csv(f"{absolute_path}/{output_path}/{uid}/random_GeO_full_distribution_{NUM_SHUFFLES}.csv")
