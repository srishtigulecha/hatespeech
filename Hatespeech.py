from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import tamilstemmer
import pandas as pd
import nltk
import string
import csv
import streamlit as st
import pickle
import tweepy
import tensorflow as tf

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('indian')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.sequence import pad_sequences

eng_stop_words = set(stopwords.words('english'))
tamil_sw_file = 'tamil-stopwords.txt'
model = tf.keras.models.load_model('new_model_hatespeech.h5')

text_data = []

with open(tamil_sw_file, 'r', encoding='utf-8') as temp_output_file:
    reader = csv.reader(temp_output_file, delimiter='\n')
    for row in reader:
        text_data.append(row)

tamil_stop_words = [x[0] for x in text_data]

def encode_text(tokenizer, lines, length):
    encoded=tokenizer.texts_to_sequences(lines)
    padded= pad_sequences(encoded, maxlen=length, padding='post')
    return padded

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

trainLength=300

st.title("Tamil Regional Language Hate Speech Detection in Twitter")


st.sidebar.subheader("About")
about="""
The rapid growth of Internet users led to unwanted cyber issues, including cyberbullying, hate speech, and many more. Twitter which is a popularly used social media platform has seen a rapid rise in hate speech tweets and comments. This project deals with the problems of hate speech on Twitter. 
This project uses a Multi layer Convolutional, BiGRU and Capsule network based deep learning model that classifies tamil tweets as offensive or not offensive.
"""
st.sidebar.write(about)
search_query = st.text_input("Enter query:")

if st.button("Detect"):
    twts = []
    tamil_stop_words="""
ஒரு என்று மற்றும் இந்த இது என்ற கொண்டு என்பது பல ஆகும் அல்லது அவர் நான் உள்ள அந்த இவர் என முதல் என்ன இருந்து சில என் போன்ற வேண்டும் வந்து இதன் அது அவன் தான் பலரும் என்னும் மேலும் பின்னர் கொண்ட இருக்கும் தனது உள்ளது போது என்றும் அதன் தன் பிறகு அவர்கள் வரை அவள் நீ ஆகிய இருந்தது உள்ளன வந்த இருந்த மிகவும் இங்கு மீது ஓர் இவை இந்தக் பற்றி வரும் வேறு இரு இதில் போல் இப்போது அவரது மட்டும் இந்தப் எனும் மேல் பின் சேர்ந்த ஆகியோர் எனக்கு இன்னும் அந்தப் அன்று ஒரே மிக அங்கு பல்வேறு விட்டு பெரும் அதை பற்றிய உன் அதிக அந்தக் பேர் இதனால் அவை அதே ஏன் முறை யார் என்பதை எல்லாம் மட்டுமே இங்கே அங்கே இடம் இடத்தில் அதில் நாம் அதற்கு எனவே பிற சிறு மற்ற விட எந்த எனவும் எனப்படும் எனினும் அடுத்த இதனை இதை கொள்ள இந்தத் இதற்கு அதனால் தவிர போல வரையில் சற்று எனக்
"""
    st.write("data related to ",search_query,"being scraped from twitter.... ")

    consumer_key = "" #Your API/Consumer key 
    consumer_secret = "" #Your API/Consumer Secret Key
    access_token = ""    #Your Access token key
    access_token_secret = "" #Your Access token Secret key


    #Pass in our twitter API authentication key
    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret,
        access_token, access_token_secret
    )

    #Instantiate the tweepy API
    api = tweepy.API(auth, wait_on_rate_limit=True)

    no_of_tweets =150

    try:
        #The number of tweets we want to retrieved from the search
        tweets = api.search_tweets(q=search_query, count=no_of_tweets)
        
        #Pulling Some attributes from the tweet
        attributes_container = [[tweet.user.name, tweet.created_at, tweet.favorite_count, tweet.source,  tweet.text] for tweet in tweets]

        #Creation of column list to rename the columns in the dataframe
        columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
        
        #Creation of Dataframe
        tweets_df = pd.DataFrame(attributes_container, columns=columns)

    except BaseException as e:
        print('Status Failed On,',str(e))

    st.dataframe(tweets_df) 
    

    res1 = {'tweets':[] , 'Offensive/Not Offensive':[]}
    cols = ['Tweet' , 'Offensive/Not Offensive']
    for i in range(len(tweets_df)):
      text = tweets_df['Tweet'].loc[i]
      restweet=tweets_df['Tweet'].loc[i]
      text = text.lower() 

      #remove punctuation
      translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
      text = text.translate(translator) 

      #tokenization
      tokens = text.split()
        
      # filter out stop words
      tokens = [w for w in tokens if not w in tamil_stop_words]
      tokens = [w for w in tokens if not w in eng_stop_words]

      #transliterate text
      for i in range(len(tokens)):
        tokens[i] = transliterate(tokens[i], sanscript.ITRANS, sanscript.TAMIL)

      # Perform stemming using Open-Tamil
      stemmer = tamilstemmer.TamilStemmer()

      lemmatizer = WordNetLemmatizer()

      for i in range(len(tokens)):
        tokens[i] =  stemmer.stemWord(tokens[i])
        tokens[i] = lemmatizer.lemmatize(tokens[i], pos='v')
        
      newtext = " ".join(tokens)
     
      newt = []
      newt.append(newtext)
      val = encode_text(tokenizer, newt, trainLength)

      sentiment = ['Not_offensive','Offensive_Targeted_Insult_Group','Offensive_Targeted_Insult_Individual','Offensive_Targeted_Insult_Other', 'Offensive_Untargetede', 'not-Tamil']
      ypred = model.predict([val])
      print(ypred)
      s=0
      for i in range(1,6):
        s+=ypred[0][i]
      if s>ypred[0][0]:
        res1['Offensive/Not Offensive']+=['Offensive']
        res1['tweets']+=[restweet]
      else:
        res1['Offensive/Not Offensive']+=['Not Offensive']
        res1['tweets']+=[restweet]
      
    res= pd.DataFrame(res1)
    st.dataframe(res) 
  
  
