import os
from bs4 import BeautifulSoup as bs
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_and_clean_reviews(directory: str) -> list[str]:
    '''
    Returns a list of text documents from a folder

    Input
    ------
    A directory containing .txt docuemnts

    Output
    -------
    A list of strings containing the text within the .txt documents
    '''
    reviews = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory,filename), 'r', encoding='utf-8') as file:
                print(file.name)
                review = file.read()
                soup = bs(review, "html.parser")
                clean_review = soup.get_text()
                reviews.append(clean_review)
    return reviews

def load_clean_reviews_to_CSV():
    '''
    Load the reviews into the csv file
    '''

    negative_review_dir = 'aclImdb/test/neg'
    positive_review_dir = 'aclImdb/test/pos'

    negReviews = load_and_clean_reviews(negative_review_dir)
    posReviews = load_and_clean_reviews(positive_review_dir)

    positive_df = pd.DataFrame(posReviews, columns=['review'])
    positive_df['label'] = 1
    
    negative_df = pd.DataFrame(negReviews, columns=['review'])
    negative_df['label'] = 0

    all_reviews_df = pd.concat([positive_df,negative_df], ignore_index=True)

    all_reviews_df.to_csv('cleaned_imdb_reviews.csv', index=False)

def preprocess_csv():
    '''
    Cleans all text in place from the review column in the csv
    '''
    df = pd.read_csv('cleaned_imdb_reviews.csv')
    df['review'] = df['review'].apply(preprocess_text)
    df.to_csv('cleaned_imdb_reviews.csv', index=False)

def preprocess_text(text: str) -> str:
    '''
    Cleans text to be tokenized. Removes puntuation, converts to lowercase, removes stopwords.
    
    Input
    ------
    Any string

    Output
    ------
    Cleaned string
    '''
    punctuation = string.punctuation

    text = text.lower()
    text = ''.join([char for char in text if char not in punctuation])

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    '''
    #This code creates and cleans all the text documents
    
    posTestReviewsDf = pd.DataFrame(load_and_clean_reviews('aclimdb/test/pos'), columns=['review'])
    posTestReviewsDf['label'] = 1
    negTestReviewsDf = pd.DataFrame(load_and_clean_reviews('aclimdb/test/neg'), columns=['review'])
    negTestReviewsDf['label'] = 0
    negTrainReviewsDf = pd.DataFrame(load_and_clean_reviews('aclImdb/train/neg'), columns =['review'])
    negTrainReviewsDf['label'] = 0
    posTrainReviewsDf = pd.DataFrame(load_and_clean_reviews('aclImdb/train/pos'), columns = ['review'])
    posTrainReviewsDf['label'] = 1
    
    df = pd.concat([posTestReviewsDf,negTestReviewsDf,posTrainReviewsDf,negTrainReviewsDf], ignore_index=True)
    df.to_csv('cleaned_imdb_reviews.csv', index=False)

    preprocess_csv()
    '''
    
if __name__ == '__main__': 
    main()