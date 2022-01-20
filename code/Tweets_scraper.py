import pandas as pd
import tweepy
import datetime
import time

# load user data
users = pd.read_csv("user_5000.csv")
user_ids = users['User ID'][0:30]


# tweet authorization
consumerKey =""
consumerSecret =""
accessToken =""
accessTokenSecret =""

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

startDate = datetime.datetime(2020, 1, 15)
endDate =   datetime.datetime(2020, 10,15)
data = {"User ID": [], "Tweet ID": [], "Tweet Text": [], "Date": [], "Time": []}
totalTime = 0
user_count = 0
tic1 = time.perf_counter()

if len(user_ids) > 0:
    for target in user_ids:
        tic = time.perf_counter()
        print("User ID: " + str(target))
        

        # Collect tweets
        tmpTweets = tweepy.Cursor(api. user_timeline, id = target).items()
        try: 
            for tweet in tmpTweets:
                #if tweet.created_at < endDate and tweet.created_at > startDate:
                if tweet.created_at < endDate and tweet.created_at > startDate and tweet.lang == "en" and tweet.text[0:2] != "RT":
                    data["Tweet Text"].append(tweet.text)
                    data["Tweet ID"].append(tweet.id)
                    date_time = str(tweet.created_at).split(" ")
                    data["Date"].append(date_time[0])
                    data["Time"].append(date_time[1])
                    data["User ID"].append(str(target))

            #user_data_frame = pd.DataFrame(data)
            toc = time.perf_counter()
            print(f"Consumed time for tweets collection: {toc - tic:0.4f} seconds")
            #print(user_data_frame.shape)
            print("=======DONE=======")
            #time.sleep(120)
            user_count += 1
            totalTime += toc-tic
        except:
            pass
toc2 = time.perf_counter()
print(f"Total consumed tweet collection time: {totalTime:0.4f} seconds.")
print(f"Total consumed running time: {toc2 - tic1: 0.4f} seconds.")
print("The number of valid users: "+str(user_count))
data_dataframe = pd.DataFrame(data=data)
data_dataframe.to_csv("output.csv")
