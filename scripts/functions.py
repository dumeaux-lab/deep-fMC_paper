import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import silhouette_score
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy
from scipy.cluster.hierarchy import cophenet, leaves_list
import fastcluster as fc

def StabilityPerArchetype(results_dict, k_min, k_max, step, n_rep, l2=False):
    
    K = np.arange(k_min, k_max + 1, step)
    stability_results = []

    for k in K:
        # Collect all decoded data for the current k
        all_arch = []
        for rs in range(0, n_rep):
            if (k, rs) in results_dict:
                decoded_data = results_dict[(k, rs)]['spectra']
                all_arch.append(decoded_data)

        if all_arch:
            # Combine all decoded data
            arch = np.concatenate(all_arch, axis=0)
            arch_df = pd.DataFrame(arch)

            if l2:
                arch_df = (arch_df.T / np.sqrt((arch_df**2).sum(axis=1))).T

            # K-means clustering on all spectra
            kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
            kmeans_model.fit(arch_df)
            kmeans_cluster_labels = pd.Series(kmeans_model.labels_ + 1, index=arch_df.index)

            # Compute the silhouette score for each sample
            silhouette_vals = silhouette_samples(arch_df.values, kmeans_cluster_labels, metric='euclidean')
            
            # Aggregate silhouette scores by cluster
            cluster_stabilities = pd.DataFrame({
                "Cluster": kmeans_cluster_labels,
                "Silhouette": silhouette_vals
            }).groupby("Cluster").mean().reset_index()
            
            # Add the number of clusters and method to the results
            cluster_stabilities["K"] = k
            
            stability_results.append(cluster_stabilities)

    # Concatenate all results into a single DataFrame
    stability_results_df_list = list(stability_results)
    
    return stability_results_df_list

def concatAnnotatedDatasets(dataset1, dataset2):
    concatenated_array = np.concatenate((dataset1, dataset2), axis=1)
    return concatenated_array

def concatDataFrameDatasets(dataset1, dataset2):
    concatenated_df = pd.concat([dataset1, dataset2], axis=0)
    return concatenated_df

def convertToAnnData(data, metadata_row):
    data_ad = ad.AnnData(data.transpose())
    data_ad.X = data_ad.X.astype(float)
    # Add the metpathways_ad row to .obs
    data_ad.obs = pd.DataFrame(metadata_row)
    # Filter out genes with non-zero expression in less than 1 cells
    sc.pp.filter_genes(data_ad, min_cells=1)

    # filter out cells that express no genes
    sc.pp.filter_cells(data_ad, min_genes=1)

    # Calculate total number of counts and total number of genes
    data_ad.obs["n_counts"] = data_ad.X.sum(axis=1) # number of expressions of a row
    data_ad.obs["n_pathways"] = (data_ad.X>0).sum(axis=1) # count the number of non-zero cells in a row

    #sc.pl.violin(data_ad, ['n_pathways', 'n_counts'], jitter=0.4, multi_panel=True, save=f"n_path_and_counts_violin.pdf") 
    sc.pl.violin(data_ad, ['n_pathways', 'n_counts'], jitter=0.4, multi_panel=True) 

    data_ad_original = data_ad.copy()
    sc.pp.normalize_total(data_ad, target_sum=1e4)
    sc.pp.log1p(data_ad)
    data_ad.raw = data_ad
    sc.pp.scale(data_ad, max_value=10)
    sc.tl.pca(data_ad, svd_solver='auto')
    sc.pl.pca_variance_ratio(data_ad, log=True, n_pcs=50)

    sc.pp.neighbors(data_ad, n_pcs=50, n_neighbors=15)
    sc.tl.umap(data_ad)
    obs_list = ['country', 'Global Region', 'study', 'n_pathways', 'n_counts']
    for i in obs_list:
        #sc.pl.umap(data_ad,color=i, save=f"umap_plot_{i}.pdf")
        sc.pl.umap(data_ad,color=i)

    return data_ad_original


def plotPCA(k, results, n_components, states):
    all_arch = []
    for rs in range(0, states):  # Example range of rs
        if (k, rs) in results:
            decoded_data = results[(k, rs)]['spectra']
            all_arch.append(decoded_data)

    # Combine all decoded data
    arch = np.concatenate(all_arch, axis=0)
    arch_df = pd.DataFrame(arch)

    # Perform K-means clustering
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(arch_df)
    kmeans_labels = kmeans_model.labels_

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    arch_pca = pca.fit_transform(arch_df)

    # Plot the K-means clustering results
    plt.figure(figsize=(10, 6))
    #print(kmeans_labels)
    # Count the occurrences of each unique value
    unique, counts = np.unique(kmeans_labels, return_counts=True)

    # Create a dictionary to display counts
    value_counts = dict(zip(unique, counts))
    print(value_counts)
    
    for i in range(k):
        cluster_points = arch_pca[kmeans_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

    '''    # Annotate points 
        for index in [45, 46, 47, 276, 277, 278, 72, 73, 74]:
            plt.annotate(f'Point {index}', (arch_pca[index, 0], arch_pca[index, 1]),
                        textcoords="offset points", xytext=(0,10), ha='center')
    '''
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"K-means Clustering of Archetypes for K={k}")
    plt.legend()
    plt.grid(True)
    plt.show()

    weightedPCA(pca)


def weightedPCA(pca):
    explained_variance_ratio = pca.explained_variance_ratio_
    weights = explained_variance_ratio

    # Compute weighted importance
    weighted_importance = weights
    weighted_importance_normalized = weighted_importance / np.sum(weighted_importance)

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'PCA Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Explained Variance Ratio': explained_variance_ratio,
        'Weighted Importance': weighted_importance,
        'Normalized Weighted Importance': weighted_importance_normalized
    })

    # Display the DataFrame
    print(importance_df)



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def StabilityArchetype(results_dict, k_min, k_max, step, n_rep, l2=False):
    '''
    Borrowed from Alexandrov et al. 2013: Deciphering signatures of mutational processes 
    operative in human cancer and Kotliar et al. 2019: Identifying gene expression programs
    of cell-type identity and cellular activity with single-cell RNA-seq
    '''
    
    K = np.arange(k_min, k_max + 1, step)
    stability_results = []
    labels_dict = {}

    for k in K:
        # Collect all decoded data for the current k
        all_arch = []
        for rs in range(n_rep):
            if (k, rs) in results_dict:
                decoded_data = results_dict[(k, rs)]['spectra']
                all_arch.append(decoded_data)

        if all_arch:
            # Combine all decoded data
            arch = np.concatenate(all_arch, axis=0)
            arch_df = pd.DataFrame(arch)

            if l2:
                arch_df = (arch_df.T / np.sqrt((arch_df**2).sum(axis=1))).T

            # K-means clustering on all spectra
            kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
            kmeans_model.fit(arch_df)
            kmeans_cluster_labels = pd.Series(kmeans_model.labels_ + 1, index=arch_df.index)

            # Store the labels for this k
            labels_dict[k] = kmeans_cluster_labels

            # Compute the silhouette score
            stability = silhouette_score(arch_df.values, kmeans_cluster_labels, metric='euclidean')
            stability_results.append([k, stability])

    stability_results_df = pd.DataFrame(stability_results, columns=["K", "Stability"])
    return stability_results_df, labels_dict

def StabilityUsage(k_min, k_max, step, n_rep, n_cells, loaded_results, filepath_consensus=None, 
                   n_cells_max=20000, cluster=True):
    
    '''
    Borrowed from Brunet et al. 2004: Metagenes and molecular pattern discovery using
    matrix factorization
    '''
    np.random.seed(520)
    
    K = np.arange(k_min, k_max+1, step)
    results = []
    
    if n_cells > n_cells_max:
        cells_choice = np.random.choice(n_cells, n_cells_max, replace=False)
        n_cells = n_cells_max
    else:
        cells_choice = np.arange(n_cells)
    
    for k in K:
            d = np.zeros(int(scipy.special.comb(n_cells, 2)))
            # Read all cell usage results
            for rs in np.arange(0, n_rep):
                usage = loaded_results[(k, rs)]['usage']
                usage = usage[cells_choice,]               
                if cluster:
                    assign = usage.argmax(1)
                    usage = np.zeros_like(usage)
                    usage[np.arange(len(usage)), assign] = 1
                    #print('usage:', usage)
                    d += scipy.spatial.distance.pdist(usage, 'braycurtis')
                    #print('d: ', d)
                else:
                    d += scipy.spatial.distance.pdist(usage)
            
            d = d/n_rep
            
            
            # Hierarchical clustering using distance d
            HC = fc.linkage(d, method='average')
            cophen_d = cophenet(HC)
            
            # Compute Cophenetic correlation coefficient
            cophen_corr = np.corrcoef(d, cophen_d)[0,1]
            results.append([k, cophen_corr])
    
    results = pd.DataFrame(results, columns=["K", "Stability"])
    
    return results
    
def extract_rows_by_label(results_dict, labels_dict, k, n_rep):
    if k not in labels_dict:
        print(f"Labels for k={k} not found in labels_dict.")
        return None
    
    kmeans_cluster_labels = labels_dict[k]
    label_dfs = {label: [] for label in range(1, k+1)}

    row_index = 0
    for rs in range(n_rep):
        if (k, rs) in results_dict:
            df = pd.DataFrame(results_dict[(k, rs)]['spectra'])
            n_rows = df.shape[0]
            for label in label_dfs:
                label_rows = df.iloc[np.where(kmeans_cluster_labels.iloc[row_index:row_index + n_rows] == label)]
                label_dfs[label].append(label_rows)
            row_index += n_rows

    # Concatenate all DataFrames for each label
    label_dfs = {label: pd.concat(label_dfs[label], axis=0) for label in label_dfs}

    return label_dfs


def extract_rows_by_label_with_usage(results_dict, labels_dict, k, n_rep, output_dir='output'):
    if k not in labels_dict:
        print(f"Labels for k={k} not found in labels_dict.")
        return None

    kmeans_cluster_labels = labels_dict[k]
    label_dfs = {label: [] for label in range(1, k+1)}

    row_index = 0
    for rs in range(n_rep):
        if (k, rs) in results_dict:
            df = pd.DataFrame(results_dict[(k, rs)]['spectra'])
            usage_df = pd.DataFrame(results_dict[(k, rs)]['usage'])
            
            n_rows = df.shape[0]
            
            # Ensure the index range is within bounds
            if row_index + n_rows > len(kmeans_cluster_labels):
                print(f"Index range out of bounds: row_index={row_index}, n_rows={n_rows}, length of labels={len(kmeans_cluster_labels)}")
                break

            # Rename columns in usage_df based on the KMeans labels
            df_columns = df.columns
            new_column_names = {i: f'type_{i}' for i in range(k)}
            
            # Create new column names mapping
            new_column_names_mapping = {}
            for i in range(len(df_columns)):
                if row_index + i < len(kmeans_cluster_labels):
                    label = kmeans_cluster_labels.iloc[row_index + i]
                    new_column_name = new_column_names.get(label, f'type_{label}')
                    new_column_names_mapping[df_columns[i]] = new_column_name

            # Rename columns in usage_df
            usage_df.rename(columns=new_column_names_mapping, inplace=True)

            # Save usage_df to CSV
            usage_filename = f'{output_dir}/usage_df_k{k}_rep{rs}.csv'
            usage_df.to_csv(usage_filename, index=False)
            print(f'Saved usage_df to {usage_filename}')

            # Collect rows by label
            for label in label_dfs:
                label_rows = df.iloc[np.where(kmeans_cluster_labels.iloc[row_index:row_index + n_rows] == label)]
                label_dfs[label].append(label_rows)

            row_index += n_rows

    # Concatenate all DataFrames for each label
    label_dfs = {label: pd.concat(label_dfs[label], axis=0) for label in label_dfs}

    return label_dfs

