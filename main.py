import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import chardet
from datetime import datetime
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud


header = st.beta_container()
dataset = st.beta_container()
visualizations = st.beta_container()
nLp = st.beta_container()

with header:
    st.title("Welcome to Manuel's Twitter Sentiment Analysis Web App")
    st.text("You must extract data first in order to proceed.")



with dataset:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("Extract data from Twitter")

    #see = pd.read_csv(r'C:\Users\29052020\Documents\Crypto\Twitter sentiment analysys\Login.csv')
    #see.to_csv(r'Login.csv')

    log = pd.read_csv(r"Login.csv")

    # Twitter API credentials
    consumerKey = log["API key"].iloc[0]
    consumerSecret = log["API Secret Key"].iloc[0]

    # Create the authentication object
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

    # Create the API object while passing in the auth information
    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    numTweets_ = st.slider("Minimum number of Tweets to Explore", min_value=10, max_value=20000)
    today = datetime.today().date()
    date_since_ = st.date_input('Date Since', today)
    st.text("Examples to fill in the search box: '#word or #letter or #mail'")
    search_words_ = st.text_input("#'s to search on Twitter: ")


    def scraptweets(search_words, date_since, numTweets):

        db_tweets = pd.DataFrame(columns=['username', 'acctdesc', 'location', 'following',
                                          'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
                                          'retweetcount', 'text', 'hashtags'])

        program_start = datetime.time()

        start_run = datetime.time()

        tweets = tweepy.Cursor(api.search, q=search_words, lang="en", since=date_since, tweet_mode='extended').items(
            numTweets)

        print("Loading")

        tweet_list = [tweet for tweet in tweets]

        reTweets = 0
        numTweets = 0
        numDuplicated = 0

        # Inicio de Loop

        for tweet in tweet_list:
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            location = tweet.user.location
            following = tweet.user.friends_count
            followers = tweet.user.followers_count
            totaltweets = tweet.user.statuses_count
            usercreatedts = tweet.user.created_at
            tweetcreatedts = tweet.created_at
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']

            # the following can be used to print the full text of the Tweet, or if it’s a Retweet, the full text of the Retweeted Tweet:

            try:
                text = tweet.retweeted_status.full_text
                reTweets += 1
                # Check-in
                if ((numTweets + reTweets + numDuplicated) % 2000) == 0:
                    print("no. of tweets scraped is {}, the number of retweets is {} and the number of duplicates is {}. Please wait 5 min for more scrapping".format(numTweets, reTweets, numDuplicated))
                    time.sleep(300)  # 5 minutos sleep time
                    continue


            except AttributeError:  # Not a Retweet
                text = tweet.full_text

            ith_tweet = [username, acctdesc, location, following, followers, totaltweets,
                         usercreatedts, tweetcreatedts, retweetcount, text, hashtags]


            # Já existe na database?

            resultado1 = None

            for x in db_tweets["username"]:
                if x == ith_tweet[0]:
                    resultado1 = True

            resultado2 = None

            for y in db_tweets["text"]:
                if y == ith_tweet[9]:
                    resultado2 = True

            # Vamos a confirmar

            if resultado1 and resultado2:
                numDuplicated += 1
                # Check-in
                if ((numTweets + reTweets + numDuplicated) % 2000) == 0:
                    print("no. of tweets scraped is {}, the number of retweets is {} and the number of duplicates is {}. Please wait 5 min for more scrapping".format(numTweets, reTweets, numDuplicated))
                    time.sleep(300)  # 5 minutos sleep time
                    continue

            else:
                db_tweets.loc[len(db_tweets)] = ith_tweet
                numTweets += 1
                # Check-in

                if ((numTweets + reTweets + numDuplicated) % 2000) == 0:
                    print("no. of tweets scraped is {}, the number of retweets is {} and the number of duplicates is {}. Please wait 5 min for more scrapping".format(numTweets, reTweets, numDuplicated))
                    time.sleep(300)  # 5 minutos sleep time

            # Check-in

            #if ((numTweets + reTweets + numDuplicated) % 2000) == 0:
                #print("no. of tweets scraped is {}, the number of retweets is {} and the number of duplicates is {}. Please wait 5 min for more scrapping".format(numTweets, reTweets, numDuplicated))
                #time.sleep(300)  # 5 minutos sleep time
                #pass




        end_run = time.time()

        duration_run = round((end_run - start_run) / 60, 2)

        print("Extraction Complete: no. of tweets scraped is {}, the number of ignored retweets is {} and the number of ignored duplicates is {}".format(numTweets, reTweets, numDuplicated))
        print("Extration was completed in {} min".format(duration_run))

        # Fim de loop
        for x in db_tweets["followers"]:
            if x == 0:
                x = 1

        db_tweets.to_csv("Twitter.csv", index=False)

        return db_tweets

    #scraptweets(search_words=search_words_, date_since=date_since_, numTweets=numTweets_)

    
    database = scraptweets(search_words=search_words_, date_since=date_since_, numTweets=numTweets_)


with visualizations:

    st.header("Graphs based on Twitter Data")

    #Data Manipulation

    database['followers'] = database['followers'].replace(0, 1)

    database["retweetsPerFollowers"] = database["retweetcount"] / database["followers"]

    #database["retweetsPerFollowers"].replace([np.inf, -np.inf], 0, inplace=True)
    database["retweetsPerFollowers"] = pd.to_numeric(database["retweetsPerFollowers"])

    #database["retweetsPerFollowers"].replace([np.inf, -np.inf], 0, inplace=True)

    database["retweetsPerFollowers"] = pd.to_numeric(database["retweetsPerFollowers"])

    database["tweetcreatedts"] = pd.to_datetime(database["tweetcreatedts"])

    dicio = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}

    database["Day of the Week"] = database["tweetcreatedts"].dt.dayofweek
    database["Day of the Week"].replace(dicio, inplace=True)

    categorias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    #cat = CategoricalDtype(categories=categorias, ordered=True)

    #database['Day of the Week'] = st.write(database["Day of the Week"].astype(cat)

    daysofweekcount = database["Day of the Week"].value_counts().sort_index()

    database["usercreatedts"] = pd.to_datetime(database["usercreatedts"])

    database["usercreatedts"] = database["usercreatedts"].dt.strftime('%Y-%m-%d')

    database["usercreatedts"] = pd.to_datetime(database["usercreatedts"])

    datahoje = []

    for x in database["usercreatedts"]:
        datahoje.append(datetime.today().date())

    database["date of extraction"] = datahoje

    database["date of extraction"] = pd.to_datetime(database["date of extraction"])

    database["howLongAccountWasCreated"] = database["date of extraction"] - database["usercreatedts"]

    database["howLongAccountWasCreated"] = database["howLongAccountWasCreated"].astype(str)

    database["howLongAccountWasCreated"] = database["howLongAccountWasCreated"].str.split(' ').str[0]

    database["howLongAccountWasCreated"] = database["howLongAccountWasCreated"].astype(int)

    # Quantos followers foram adicionados por dia

    database["followersPerDay"] = database["followers"] / database["howLongAccountWasCreated"]

    database["followersPerDay"] = database["followersPerDay"].astype(float)

    user_index = database.set_index("username")

    top_ret_f = user_index.sort_values(["retweetsPerFollowers"], ascending=False).head(10)

    top_fol_hl = user_index.sort_values(["followersPerDay"], ascending=False).head(10)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html

    database
    #corre = database.corr()
    #mask = np.zeros_like(corre)
    #mask[np.triu_indices_from(mask)] = True

    #fig, ax = plt.subplots(figsize=(12, 4))
    #sns.heatmap(corre, cmap="YlGnBu", annot=True, linewidths=.1, mask=mask, vmin=0, vmax=1, ax=ax)
    #st.pyplot(fig)

    #plt.title('Correlation')

    st.text("Top users of Retweets per Followers")

    top_ret_f = top_ret_f[~top_ret_f.index.duplicated(keep='first')]
    top_ret_f_items = top_ret_f[["retweetsPerFollowers"]]
    st.bar_chart(top_ret_f_items)

    st.text(top_ret_f_items.index)

    st.text("Tweets on day of the Week")

    st.line_chart(daysofweekcount)

    st.text("Top users of Followers Per Day")

    top_fol_hl = top_fol_hl[~top_fol_hl.index.duplicated(keep='first')]
    top_fol_hl_items = top_fol_hl[["followersPerDay"]]
    st.bar_chart(top_fol_hl_items)

    st.text(top_fol_hl_items.index)

with nLp:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("Finally! Sentiment Analysis")
    st.text("Did the result meet your expectations?")


    def cleanTxt(text):
        text = re.sub(r'@[A-Za-z0-9]+', "", text)  # Removed @mentions
        text = re.sub(r'"#', "", text)  # Removing the "#" symbol
        text = re.sub(r'RT[\s]+', "", text)  # Removing RT
        text = re.sub(r'https?:\/\/\S+', "", text)  # Remove the hyper link

        return text


    # Cleaning the text

    database["text"] = database["text"].apply(cleanTxt)


    # Create a function to get the subjectivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity


    # Create a function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity


    # Create two new columns
    database["Subjectivity"] = database["text"].apply(getSubjectivity)
    database["Polarity"] = database["text"].apply(getPolarity)

    # Plot the Word Cloud

    allWords = " ".join([twts for twts in database["text"]])
    wordCloud = WordCloud(width=600, height=400, random_state=21, max_font_size=119).generate(allWords)

    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()


    # Create a function to compute the negative, neutral and positive analysis

    def getAnalysis(score):
        if score < 0:
            return "Negative"
        elif score == 0:
            return "Neutral"
        else:
            return "Positive"


    database["Analysis"] = database["Polarity"].apply(getAnalysis)

    # Plot the polarity and subjectivity
    plt.figure(figsize=(8, 6))
    for i in range(0, database.shape[0]):
        plt.scatter(database["Polarity"][i], database["Subjectivity"][i], color="Blue")

    plt.title("Sentiment Analysis")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    st.pyplot()

    # Get the percentage of positive tweets
    ptweets = database[database.Analysis == "Positive"]
    ptweets = ptweets["text"]

    st.text("Positive tweets = " + str(round((ptweets.shape[0] / database.shape[0]) * 100, 1)) + " %")

    # Get the percentage of neutral tweets
    ptweets = database[database.Analysis == "Neutral"]
    ptweets = ptweets["text"]

    st.text("Neutral tweets = " + str(round((ptweets.shape[0] / database.shape[0]) * 100, 1)) + " %")

    # Get the percentage of negative tweets
    ptweets = database[database.Analysis == "Negative"]
    ptweets = ptweets["text"]

    st.text("Negative tweets = " + str(round((ptweets.shape[0] / database.shape[0]) * 100, 1)) + " %")

    # Show the value counts

    database["Analysis"].value_counts()

    # plot and visualize

    plt.title("Sentiment Analysis")
    plt.xlabel("Counts")
    plt.ylabel("Sentiment")
    database["Analysis"].value_counts().plot(kind="barh")
    st.pyplot()

