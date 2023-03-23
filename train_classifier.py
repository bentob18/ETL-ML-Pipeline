import sys
import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
nltk.download('punkt')
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: Training message List.
        Y: Training target.
        category_names: Categorical name for labeling.
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('FigureEight', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    categories = df.columns[4:].tolist()
    return X, Y, categories

def tokenize(message):
    '''
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Result list after tokenization.
    '''
    
    text = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    input:
        None
    output:
        cv: GridSearch model result.
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
    'clf__estimator__n_estimators': [10, 20, 30],
    'clf__estimator__max_depth': [None, 10, 20],
    'clf__estimator__min_samples_split': [2, 3, 4],
    'clf__estimator__criterion': ['gini', 'entropy']
}
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)
    return grid_search

def evaluate_model(model, X_test, categories):
    y_pred_df = pd.DataFrame(y_pred, columns=categories)
    y_test_df = pd.DataFrame(y_test, columns=categories)
    y_test_sel = y_test_df.loc[:, y_pred_df.columns]

for category in categories:
    print("Category: {}".format(category))
    print(classification_report(y_test_sel[category], y_pred_df[category]))

def save_model(model, model_filepath):
     joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
