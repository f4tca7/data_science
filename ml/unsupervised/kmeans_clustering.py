import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

df_grain = pd.read_csv('../datasets/seeds.csv')
samples_grain = df_grain.iloc[:,:7].values
varieties_grain = df_grain.iloc[:,7:]
varieties_grain_str = varieties_grain.copy()
varieties_grain_numbers = varieties_grain.values.reshape(-1)
varieties_grain_str.loc[varieties_grain.iloc[:, 0] == 1] = 'Kama wheat'
varieties_grain_str.loc[varieties_grain.iloc[:, 0] == 2] = 'Rosa wheat' 
varieties_grain_str.loc[varieties_grain.iloc[:, 0] == 3] = 'Canadian wheat'
varieties_grain_arr = varieties_grain_str.values.reshape(-1)

df_fish = pd.read_csv('../datasets/fish.csv')
samples_fish = df_fish.iloc[:,1:7].values
species_fish = df_fish.iloc[:,:1].values.reshape(-1)

df_stock = pd.read_csv('../datasets/company-stock-movements-2010-2015-incl.csv')
movements = df_stock.iloc[:,1:].values
companies = df_stock.iloc[:,:1].values.reshape(-1)


df_esc = pd.read_csv('../datasets/eurovision-2016.csv')
numeric_vals = df_esc.iloc[:, 2:].values
imp_mean = Imputer(missing_values=np.nan, strategy='mean', axis=1)
imp_mean.fit(numeric_vals)
numeric_vals = imp_mean.transform(numeric_vals)
df_esc.iloc[:,2:] = numeric_vals
df_esc.iloc[:,2:] = df_esc.iloc[:,2:].astype('int64', copy=False)
col = df_esc.iloc[: , 2:]
df_esc['mean_vote'] = col.mean(axis=1)
cols_to_drop = [2,3,4,5,6,7,8,9,10]
df_esc.drop(df_esc.columns[cols_to_drop],axis=1,inplace=True)
esc_countries = df_esc['From country'].unique()
df_esc = df_esc.pivot(index='From country', columns='To country', values='mean_vote')
df_esc = df_esc.fillna(0)
esc_votes = df_esc.iloc[:, :].values

df_seeds_w_l = pd.read_csv('../datasets/seeds-width-vs-length.csv')
seeds_w_l_samples = df_seeds_w_l.values

##### Compare intertia for different ks

def explore_inertia(samples):
    ks = range(1, 6)
    inertias = []

    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(samples)
        
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
        
    # Plot ks vs inertias
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()

def scale_fit_crosstab(samples, categories):
    # Create scaler: scaler
    scaler = StandardScaler()

    # Create KMeans instance: kmeans
    kmeans = KMeans(n_clusters=4)

    # Create pipeline: pipeline
    pipeline = make_pipeline(scaler, kmeans)
    # Fit the pipeline to samples
    pipeline.fit(samples)

    # Calculate the cluster labels: labels
    labels = pipeline.predict(samples)

    # Create a DataFrame with labels and species as columns: df
    df = pd.DataFrame({'labels': labels, 'species': categories})

    # Create crosstab: ct
    ct = pd.crosstab(df['labels'], df['species'])
    # Display ct
    print(ct)


def clustering_stock(samples, categories) :
    # Create a normalizer: normalizer
    normalizer = Normalizer()

    # Create a KMeans model with 10 clusters: kmeans
    kmeans = KMeans(n_clusters=10)

    # Make a pipeline chaining normalizer and kmeans: pipeline
    pipeline = make_pipeline(normalizer, kmeans)

    # Fit pipeline to the daily price movements
    pipeline.fit(samples)
    # Predict the cluster labels: labels
    labels = pipeline.predict(samples)

    # Create a DataFrame aligning labels and companies: df
    df = pd.DataFrame({'labels': labels, 'companies': categories})

    # Display df sorted by cluster label
    print(df.sort_values('labels'))

##### Hierarchical clustering of the grain data

def grain_denogram(samples, varieties):
    # Calculate the linkage: mergings
    mergings = linkage(samples, method='complete')

    # Plot the dendrogram, using varieties as labels
    dendrogram(mergings,
            labels=varieties,
            leaf_rotation=90,
            leaf_font_size=6,
    )
    plt.show()

def stocks_denogram(samples, varieties) :
    # Normalize the movements: normalized_movements
    normalized_movements = normalize(samples)

    # Calculate the linkage: mergings
    mergings = linkage(normalized_movements, method='complete')

    # Plot the dendrogram
    dendrogram(mergings,
            labels=varieties,
            leaf_rotation=90,
            leaf_font_size=6
    )
    plt.show()

def esc_denogram(votes, countries) :
    # Calculate the linkage: mergings
    mergings = linkage(votes, method='single')

    # Plot the dendrogram
    dendrogram(mergings,
                labels=countries,
                leaf_rotation=90,
                leaf_font_size=6)
    plt.show()

def grain_cluster_labels(samples, varieties):
    
    mergings = linkage(samples, method='complete')
    # Use fcluster to extract labels: labels
    labels = fcluster(mergings, 6, criterion='distance')

    # Create a DataFrame with labels and varieties as columns: df
    df = pd.DataFrame({'labels': labels, 'varieties': varieties})

    # Create crosstab: ct
    ct = pd.crosstab(df['labels'], df['varieties'])

    # Display ct
    print(ct)    

def grain_TSNE(samples, variety_numbers):
    # Create a TSNE instance: model
    model = TSNE(learning_rate=200)

    # Apply fit_transform to samples: tsne_features
    tsne_features = model.fit_transform(samples)

    # Select the 0th feature: xs
    xs = tsne_features[:,0]

    # Select the 1st feature: ys
    ys = tsne_features[:,1]

    # Scatter plot, coloring by variety_numbers
    plt.scatter(xs, ys, c=variety_numbers)
    plt.show()

def stock_TSNE(samples, companies):

    normalized_movements = normalize(samples)
    # Create a TSNE instance: model
    model = TSNE(learning_rate=50)

    # Apply fit_transform to normalized_movements: tsne_features
    tsne_features = model.fit_transform(normalized_movements)

    # Select the 0th feature: xs
    xs = tsne_features[:,0]

    # Select the 1th feature: ys
    ys = tsne_features[:,1]

    # Scatter plot
    plt.scatter(xs, ys, alpha=0.5)

    # Annotate the points
    for x, y, company in zip(xs, ys, companies):
        plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
    plt.show()

def grain_pca(grains):
    # Assign the 0th column of grains: width
    model = PCA()

    # Apply the fit_transform method of model to grains: pca_features
    pca_features = model.fit_transform(grains)

    # Assign 0th column of pca_features: xs
    xs = pca_features[:,0]

    # Assign 1st column of pca_features: ys
    ys = pca_features[:,1]

    # Scatter plot xs vs ys
    plt.scatter(xs, ys)
    plt.axis('equal')
    plt.show()

    # Calculate the Pearson correlation of xs and ys
    correlation, pvalue = pearsonr(xs, ys)

    # Display the correlation
    print(correlation)

    plt.scatter(grains[:,0], grains[:,1])

    # Create a PCA instance: model
    model = PCA()

    # Fit model to points
    model.fit(grains)

    # Get the mean of the grain samples: mean
    mean = model.mean_

    # Get the first principal component: first_pc
    first_pc = model.components_[0,:]

    # Plot first_pc as an arrow, starting at mean
    plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

    # Keep axes on same scale
    plt.axis('equal')
    plt.show()     

def fish_pca(samples):
    # Create scaler: scaler
    scaler = StandardScaler()

    # Create a PCA instance: pca
    pca = PCA()

    # Create pipeline: pipeline
    pipeline = make_pipeline(scaler, pca)

    # Fit the pipeline to 'samples'
    pipeline.fit(samples)

    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()      

# explore_inertia(samples_grain)
# scale_fit_crosstab(samples_fish, species_fish)
# clustering_stock(movements, companies)
# grain_denogram(samples_grain, varieties_grain_arr)
# stocks_denogram(movements, companies)
# esc_denogram(esc_votes, esc_countries)
# grain_cluster_labels(samples_grain, varieties_grain_arr)
# grain_TSNE(samples_grain, varieties_grain_numbers)
# stock_TSNE(movements, companies)
# grain_pca(seeds_w_l_samples)
# fish_pca(samples_fish)