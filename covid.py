import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import neattext as nt
import neattext.functions as nfx

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
pd.reset_option("max_columns")

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

import warnings

st.title('COVID ANALYSIS')
st.sidebar.write('Welcome to my page')


#DATA_URL = ('C:\Users\abideen.muhammed\Downloads\vaccine\complete_vaccine_data.csv')
from sklearn.preprocessing import LabelEncoder
    

def load_data():
    data = pd.read_csv("covid.csv")
    data.fillna(method='ffill', inplace=True)
    #remove currency
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_currency_symbols)
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_stopwords)
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_numbers)
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_currencies)
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_bad_quotes)
    data["Reasons for Hesitation"]=data["Reasons for Hesitation"].apply(nfx.remove_special_characters)

    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_currency_symbols)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_stopwords)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_numbers)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_currencies)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_bad_quotes)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_bad_quotes)
    data["Obstacles"]=data["Obstacles"].apply(nfx.remove_special_characters)


    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_currency_symbols)
    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_stopwords)
    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_numbers)
    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_currencies)
    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_bad_quotes)
    data["Accessibility "]=data["Accessibility "].apply(nfx.remove_special_characters)

    #converting to numbers

    data["Accessibility_n"] = lb.fit_transform(data["Accessibility "])
    data["Obstacles_n"] = lb.fit_transform(data["Obstacles"])
    data["Reasons for Hesitation_n"] = lb.fit_transform(data["Reasons for Hesitation"])
    data["Respondent_n"] = lb.fit_transform(data["Respondent"])
    data["Race_n"] = lb.fit_transform(data["Race"])
    data["Confidence Level_n"] = lb.fit_transform(data["Confidence Level"])
    data["Trust Level_n"] = lb.fit_transform(data["Trust Level"])
    data["Safety Level_n"] = lb.fit_transform(data["Safety Level"])

    return data


df = load_data()
#st.write(df)


tf = TfidfVectorizer(max_features = 2500)
x = tf.fit_transform(df['Obstacles']).toarray()
y = df.iloc[:, 3].values



def polarity(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol < 0:
        result = "negative"
    elif pol == 0:
        result = "neutral"
    else:
        result = "positive"
    return result

df['pol']  = df["Reasons for Hesitation"].apply(polarity)


covid_df = df[df['pol'] != 'neutral']

# st.write(covid_df)

#converting to numbers
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

covid_df["pol"] = lb.fit_transform(covid_df["pol"])


X = tf.fit_transform(covid_df['Reasons for Hesitation']).toarray()
y = covid_df['pol']



# splitting the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train2, x_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.25, random_state = 40)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train2, y_train2)

y_pred = svm.predict(x_test2)

sentiment_label = df["Reasons for Hesitation"].factorize()
tokenizer = Tokenizer(num_words=5000)
covid = covid_df['Reasons for Hesitation'].values
encoded_docs = tokenizer.texts_to_sequences(covid)
padded_sequence = pad_sequences(encoded_docs, maxlen=5000)



text = st.text_input('Enter the text you want to predict: ')

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=752)
    prediction = int(svm.predict(tw).round().item())
    #prediction = int(tr.predict(tw).round().item())
    if prediction == 0:
        return 'Negative. Not like to take the vaccine'
    else:
        return 'Positive. Likey to take the vaccine'
    #print("Predicted label: ", sentiment_label[1][prediction])



st.write(f"Your text: {text} is: {predict_sentiment(text)}")


# the footer and more information
st.info("Thanks for using this app.")
