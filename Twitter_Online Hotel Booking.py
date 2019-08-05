#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import json
from datetime import datetime


# In[3]:


consumer_secret = "___"
consumer_key = "___"

access_token = "___"
access_token_secret = "___"


# In[5]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# In[6]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[8]:


target_users = ["@makemytrip", "@ibibodotcom", "@trivago", "@oyorooms", "@Cleartrip"]
sentiments = []

for target in target_users:
    print(target)
    counter = 0
    
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    
    public_tweets = api.user_timeline(target, count=200)
    
    #loop through each tweet
    for tweet in public_tweets:
        
        results = analyzer.polarity_scores(tweet["text"])
        compound = results["compound"]
        pos = results["pos"]
        neu = results["neu"]
        neg = results["neg"]
        tweets_ago = counter
        
        sentiments.append({"Tweet": tweet["text"], "Online Booking Co": target, "Tweets Ago": counter, "Date": tweet["created_at"],
                           "Compound": compound, "Positive": pos, "Negative": neg, "Neutral": neu})
        
        counter = counter +1


# In[12]:


sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd = sentiments_pd[['Online Booking Co', 'Tweets Ago', 'Date', 'Tweet', 'Compound', 'Positive', 'Neutral', 'Negative']]


# In[14]:


sentiments_pd.to_csv("Online Booking_twitter_sentiments.csv", encoding = "utf-8", index=False)


# In[15]:


orgs_colors_dict = {'@makemytrip':'blue','@ibibodotcom': 'green','@trivago': 'red','@oyorooms': 'yellow','@Cleartrip': 'purple'}

plt.scatter(sentiments_pd.groupby(["Online Booking Co"]).get_group("@makemytrip")["Tweets Ago"],
                sentiments_pd.groupby(["Online Booking Co"]).get_group("@makemytrip")["Compound"],
                  facecolors=orgs_colors_dict['@makemytrip'], edgecolors='black', label="Makemytrip")
plt.scatter(sentiments_pd.groupby(["Online Booking Co"]).get_group("@ibibodotcom")["Tweets Ago"],
                sentiments_pd.groupby(["Online Booking Co"]).get_group("@ibibodotcom")["Compound"],
                  facecolors=orgs_colors_dict['@ibibodotcom'], edgecolors='black', label="IBIBO")
plt.scatter(sentiments_pd.groupby(["Online Booking Co"]).get_group("@trivago")["Tweets Ago"],
                sentiments_pd.groupby(["Online Booking Co"]).get_group("@trivago")["Compound"],
                  facecolors=orgs_colors_dict['@trivago'], edgecolors='black', label="Trivago")
plt.scatter(sentiments_pd.groupby(["Online Booking Co"]).get_group("@oyorooms")["Tweets Ago"],
                sentiments_pd.groupby(["Online Booking Co"]).get_group("@oyorooms")["Compound"],
                  facecolors=orgs_colors_dict['@oyorooms'], edgecolors='black', label="OYO Rooms")
plt.scatter(sentiments_pd.groupby(["Online Booking Co"]).get_group("@Cleartrip")["Tweets Ago"],
                sentiments_pd.groupby(["Online Booking Co"]).get_group("@Cleartrip")["Compound"],
                  facecolors=orgs_colors_dict['@Cleartrip'], edgecolors='black', label="Cleartrip")

now = datetime.now()
now = now.strftime("%m/%d/%y")
plt.title(f'Sentiment Analysis of Online Booking Tweets ({now})')
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")

plt.xlim(100, 0)
plt.ylim(-1.0, 1.0)
yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
plt.yticks(yticks)

plt.legend(title="Online Booking Sources", bbox_to_anchor=(1, 1), frameon=False)

plt.savefig("sentiment_analysis_of_Online_booking.png")
plt.show()


# In[21]:


x_axis = np.arange(sentiments_pd["Online Booking Co"].nunique())
tick_locations = [value+0.4 for value in x_axis]

plt.title(f'Overall Online Booking Sentiment based on Twitter ({now})')
plt.xlabel("Online Booking Sources")
plt.ylabel("Tweet Polarity")

plt.bar(x_axis, sentiments_pd.groupby("Online Booking Co"). mean()["Compound"],
        color=orgs_colors_dict.values(), align="edge", width=1)
plt.xticks(tick_locations, sentiments_pd["Online Booking Co"].unique(), rotation=90)

plt.savefig("bar_sentiment.png")
plt.show()


# In[22]:


get_ipython().system('pip install wordcloud')


# In[28]:


df = pd.read_csv("Online Booking_twitter_sentiments.csv")

df.head(3)


# In[33]:


from wordcloud import WordCloud, STOPWORDS
from subprocess import check_output


# In[34]:


import matplotlib as mpl


# In[35]:


mpl.rcParams['font.size']=12
mpl.rcParams['savefig.dpi']=100
mpl.rcParams['figure.subplot.bottom']=.1


# In[37]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='white', 
                     stopwords=stopwords, max_words=200, max_font_size=40,random_state=42).generate(str(df['Tweet']))


# In[40]:


print(wordcloud)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("Online Booking", dp=1000)


# In[ ]:




