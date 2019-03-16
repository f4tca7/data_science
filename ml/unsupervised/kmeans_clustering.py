import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, MaxAbsScaler, normalize, Imputer, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


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

df_artists = pd.read_csv('../datasets/artists/artists.csv', header=None)
df_scrobbler = pd.read_csv('../datasets/artists/scrobbler-small-sample.csv')
#df_scrobbler['artist'] = df_scrobbler.loc[:, 'artist_offset'].apply(lambda x : df_artists.iloc[x])
#df_scrobbler= df_scrobbler.drop('artist_offset', axis=1)
df_scrobbler = df_scrobbler.pivot(index='artist_offset', columns='user_offset', values='playcount')
df_scrobbler = df_scrobbler.fillna(0)


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

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

    ##### Dimension reduction
    scaled_samples = scaler.fit_transform(samples)

    # Create a PCA model with 2 components: pca
    pca = PCA(n_components=2)

    # Fit the PCA instance to the scaled samples
    pca.fit(scaled_samples)

    # Transform the scaled samples: pca_features
    pca_features = pca.transform(scaled_samples)

    # Print the shape of pca_features
    print(pca_features.shape)

##### Word frequency
def word_freq():
    documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
    # Create a TfidfVectorizer: tfidf
    tfidf = TfidfVectorizer() 

    # Apply fit_transform to document: csr_mat
    csr_mat = tfidf.fit_transform(documents)

    # Print result of toarray() method
    print(csr_mat.toarray())

    # Get the words: words
    words = tfidf.get_feature_names()

    # Print words
    print(words)
    
def wiki_cluster():
    df = pd.read_csv('../datasets/wikipedia/wikipedia-vectors.csv', index_col=0)
    articles = csr_matrix(df.transpose())
    titles = list(df.columns)
    # Create a TruncatedSVD instance: svd
    svd = TruncatedSVD(n_components=50)

    # Create a KMeans instance: kmeans
    kmeans = KMeans(n_clusters=6)

    # Create a pipeline: pipeline
    pipeline = make_pipeline(svd, kmeans)
    # Fit the pipeline to articles
    pipeline.fit(articles)

    # Calculate the cluster labels: labels
    labels = pipeline.predict(articles)

    # Create a DataFrame aligning labels and titles: df
    df = pd.DataFrame({'label': labels, 'article': titles})

    # Display df sorted by cluster label
    print(df.sort_values('label'))

def wiki_nmf():
    df = pd.read_csv('../datasets/wikipedia/wikipedia-vectors.csv', index_col=0)
    articles = csr_matrix(df.transpose())
    titles = list(df.columns)    
    # Create an NMF instance: model
    model = NMF(n_components=6)

    # Fit the model to articles
    model.fit(articles)

    # Transform the articles: nmf_features
    nmf_features = model.transform(articles)

    # Print the NMF features
    print(nmf_features)   
    # Create a pandas DataFrame: df
    df = pd.DataFrame(data=nmf_features, index=titles)

    # Print the row for 'Anne Hathaway'
    print(df.loc['Anne Hathaway'])

    # Print the row for 'Denzel Washington'
    print(df.loc['Denzel Washington'])

    words_df = pd.read_csv('../datasets/wikipedia/wikipedia-vocabulary-utf8.txt', header=None)

    words = words_df.values.reshape(-1)
    print(words_df.head())
    # Create a DataFrame: components_df
    components_df = pd.DataFrame(model.components_, columns=words)
    
    # Print the shape of the DataFrame
    print(components_df.shape)

    # Select row 3: component
    component = components_df.iloc[3,:]

    # Print result of nlargest
    print(component.nlargest())

def wiki_nmf_similarity():
    df = pd.read_csv('../datasets/wikipedia/wikipedia-vectors.csv', index_col=0)
    articles = csr_matrix(df.transpose())
    titles = list(df.columns)    
    words_df = pd.read_csv('../datasets/wikipedia/wikipedia-vocabulary-utf8.txt', header=None)
    # Create an NMF instance: model
    model = NMF(n_components=6)

    # Fit the model to articles
    model.fit(articles)

    # Transform the articles: nmf_features
    nmf_features = model.transform(articles)    
    # Normalize the NMF features: norm_features
    norm_features = normalize(nmf_features)

    # Create a DataFrame: df
    df = pd.DataFrame(norm_features, index=titles)

    # Select the row corresponding to 'Cristiano Ronaldo': article
    article = df.loc['Cristiano Ronaldo']

    # Compute the dot products: similarities
    similarities = df.dot(article)

    # Display those with the largest cosine similarity
    print(similarities.nlargest())

def artist_recommendation(data, names):
    print(data.head())
    artist_names = names.values.reshape(-1)
    csr = csr_matrix(data.transpose())
    print(csr.todense())
    # Create a MaxAbsScaler: scaler
    scaler = MaxAbsScaler()

    # Create an NMF model: nmf
    nmf = NMF(n_components=20)

    # Create a Normalizer: normalizer
    normalizer = Normalizer()

    # Create a pipeline: pipeline
    pipeline = make_pipeline(scaler, nmf, normalizer)

    # Apply fit_transform to artists: norm_features
    norm_features = pipeline.fit_transform(data)    
    # Create a DataFrame: df
    df = pd.DataFrame(norm_features, index=artist_names)

    # Select row of 'Bruce Springsteen': artist
    artist = df.loc['2Pac']

    # Compute cosine similarities: similarities
    similarities = df.dot(artist)

    # Display those with highest cosine similarity
    print(similarities.nlargest())

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
# wiki_cluster()
# wiki_nmf()
# wiki_nmf_similarity()
artist_recommendation(df_scrobbler, df_artists)