{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import pandas as pd\
import re, string\
from nltk.corpus import stopwords\
from nltk.tag import pos_tag\
from nltk.stem.wordnet import WordNetLemmatizer\
import functools\
import time\
\
\
stop_words = stopwords.words('english')\
hate = pd.read_csv("hate.csv")\
counterhate = pd.read_csv("counterhate.csv")\
neutral = pd.read_csv("neutral.csv")\
data = pd.concat([hate, counterhate, neutral])\
data = data.reset_index()\
\
def remove_noise(tweet_tokens, stop_words = ()):\
\
    cleaned_tokens = []\
\
    for token, tag in pos_tag(tweet_tokens):\
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|'\\\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+',"", token)\
        token = re.sub("(@[A-Za-z0-9_]+)","", token)\
\
        if tag.startswith("NN"):\
            pos = 'n'\
        elif tag.startswith('VB'):\
            pos = 'v'\
        else:\
            pos = 'a'\
\
        lemmatizer = WordNetLemmatizer()\
        token = lemmatizer.lemmatize(token, pos)\
\
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:\
            cleaned_tokens.append(token.lower())\
    return cleaned_tokens\
\
\
filter_list = [["trump", "donald"], ["biden", "joe"], ["democratic"], ["republican"]]\
headline_list = ["Trump", "Biden", "Democratic", "Republican"]\
\
dfs = []\
\
for item in headline_list:\
    \
    index_list = []\
    for i in range (0, len(data["Tweet Text"])):\
        if any(x.lower() in filter_list[headline_list.index(item)] for x in str(data["Tweet Text"][i]).split(" ")):\
            index_list.append(i)\
\
    cleaned_data = data.loc[index_list]\
    dfs.append(cleaned_data)\
\
merge = pd.concat(dfs)\
merge = merge.drop_duplicates(subset=["Tweet Text"])\
print("Related tweets: ",merge.shape)\
merge.to_csv("output.csv")\
}