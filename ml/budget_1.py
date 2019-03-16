import numpy as np
import pandas as pd
import multilabel as multi
import sparse_interactions as sp_int
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import classification_report 
from sklearn.feature_extraction.text import HashingVectorizer


NUMERIC_COLUMNS = ['FTE', 'Total']
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']

# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # Replace nans with blanks
    text_data.fillna(' ', inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

def simple_exploration(df):
    temp_df = df
    temp_df.loc[:,'FTE'] = df.loc[:,'FTE'].apply(lambda x : np.nan if x > 1 else x)
    print(temp_df.head())
    plt.hist(df['FTE'].dropna())

    # # Add title and labels
    plt.title('Distribution of %full-time \n employee works')
    plt.xlabel('% of full-time')
    plt.ylabel('num employees')

    # Display the histogram
    plt.show()

    LABELS = ['Function',
    'Use',
    'Sharing',
    'Reporting',
    'Student_Type',
    'Position_Type',
    'Object_Type',
    'Pre_K',
    'Operating_Status']

    # Define the lambda function: categorize_label
    categorize_label = lambda x: x.astype('category')

    # Convert df[LABELS] to a categorical type
    df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

    # Print the converted dtypes
    print(df[LABELS].dtypes)

    # Calculate number of unique values for each label: num_unique_labels
    num_unique_labels = df[LABELS].apply(pd.Series.nunique)

    # Plot number of unique values for each label
    num_unique_labels.plot(kind='bar')

    # Label the axes
    plt.xlabel('Labels')
    plt.ylabel('Number of unique values')

    # Display the plot
    plt.show()

def simple_log_reg(df, holdout):

    df.loc[:,'FTE'] = df.loc[:,'FTE'].apply(lambda x : np.nan if x > 1 else x)    
    # Create the new DataFrame: numeric_data_only
    numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)
    # Get labels and convert to dummy variables: label_dummies
    label_dummies = pd.get_dummies(df[LABELS])
    print(label_dummies.head())

    # Create training and test sets
    X_train, X_test, y_train, y_test = multi.multilabel_train_test_split(numeric_data_only,
                                                                label_dummies,
                                                                size=0.2, 
                                                                min_count=2,
                                                                seed=123)

    # Print the info
    # print("X_train info:")
    # print(X_train.info())
    # print("\nX_test info:")  
    # print(X_test.info())
    # print("\ny_train info:")  
    # print(y_train.info())
    # print("\ny_test info:")  
    # print(y_test.info()) 

    # Instantiate the classifier: clf
    clf = OneVsRestClassifier(LogisticRegression())

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Print the accuracy
    print("Accuracy: {}".format(clf.score(X_test, y_test)))   

    # Load the holdout data: holdout

    holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)

    # Generate predictions: predictions
    predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS])    
    prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)

def count_text_col_tokens(df):
    # Import the CountVectorizer

    # Create the basic token pattern
    TOKENS_BASIC = '\\S+(?=\\s+)'

    # Create the alphanumeric token pattern
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Instantiate basic CountVectorizer: vec_basic
    vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

    # Instantiate alphanumeric CountVectorizer: vec_alphanumeric
    vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Create the text vector
    text_vector = combine_text_columns(df)

    # Fit and transform vec_basic
    vec_basic.fit_transform(text_vector)

    # Print number of tokens of vec_basic
    print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

    # Fit and transform vec_alphanumeric
    vec_alphanumeric.fit_transform(text_vector)

    # Print number of tokens of vec_alphanumeric
    print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))    

def simple_pipeline(df, holdout):
    # Get the dummy encoding of the labels
    dummy_labels = pd.get_dummies(df[LABELS])
    NON_LABELS = [c for c in df.columns if c not in LABELS]
    chi_k = 300

    # Split into training and test sets
    X_train, X_test, y_train, y_test = multi.multilabel_train_test_split(df[NON_LABELS],
                                                                dummy_labels,
                                                                0.2, 
                                                                min_count=3,
                                                                seed=123)

    # Preprocess the text data: get_text_data
    get_text_data = FunctionTransformer(combine_text_columns, validate=False)

    # Preprocess the numeric data: get_numeric_data
    get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)    
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Complete the pipeline: pl
    pl = Pipeline([
            ('union', FeatureUnion(
                transformer_list = [
                    ('numeric_features', Pipeline([
                        ('selector', get_numeric_data),
                        ('imputer', SimpleImputer())
                    ])),
                    ('text_features', Pipeline([
                        ('selector', get_text_data),
                        ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                        non_negative=True, norm=None, binary=False,
                                                        ngram_range=(1, 2))),
                        ('dim_red', SelectKBest(chi2, chi_k))
                    ]))
                ]
            )),
            ('int', sp_int.SparseInteractions(degree=2)),
            ('scale', MaxAbsScaler()),
            ('clf', OneVsRestClassifier(LogisticRegression()))
        ])
    # Fit to the training data
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    # Compute and print accuracy
    accuracy = pl.score(X_test, y_test)
    print("\nAccuracy on budget dataset: ", accuracy)
    print(classification_report(y_test, y_pred)) 

df = pd.read_csv('../datasets/school_budgets/TrainingData.csv', index_col=0)
holdout = pd.read_csv('../datasets/school_budgets/TestData.csv', index_col=0)
#simple_log_reg(df, holdout)

simple_pipeline(df, holdout)