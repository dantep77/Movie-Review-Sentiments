import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv('cleaned_imdb_reviews.csv')

def convertToFeaturesAndSplit():
    '''
    Creates and stores a TF-IDF vectorizer and transforms the review column of the dataframe into features.
    Splits the dataframe into testing and training sets.

    Output
    -------
    a list containing the following:
    [x training set, x testing set, y training set, y testing set]
    '''
    #Convert to TF-IDF features
    tfidf = TfidfVectorizer(encoding='utf-8', max_features=5000)
    x = tfidf.fit_transform(df['review'])
    y = df['label']
    joblib.dump(tfidf,'TFIDF-Vectorizer.pkl')

    #split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42) 
    return [x_train, x_test, y_train, y_test]

def train(splits:list):
    '''
    Trains a logarithmic regression model based on given data splits

    Input
    ------
    the output from covertToFeaturesAndSplit

    Output
    -------
    Saves the trained model to the folder
    Nothing is returned
    '''
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(splits[0],splits[2])
    
    y_pred = logreg.predict(splits[1])
    accuracy = accuracy_score(splits[3], y_pred)
    print(f'Accuracy Score: {accuracy}')
    print(classification_report(splits[3],y_pred))
    joblib.dump(logreg,'Positive-Or-Negative-Log-Reg-Model.pkl')

def main():
    '''
    splits = convertToFeaturesAndSplit()
    train(splits)
    '''


if __name__ == '__main__': 
    main()