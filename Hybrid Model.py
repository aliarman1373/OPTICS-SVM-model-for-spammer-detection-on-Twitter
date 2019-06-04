#Import Libraries
#-----------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib as plt
import collections as cl
import string
from tqdm import tqdm

import nltk.stem
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from optics import OPTICS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import statistics


#Global Variables
#-----------------------------------------------------------------------------------------
# initialise WordNet lemmatizer and punctuation filter
lemmatizer = WordNetLemmatizer()
punct = set(string.punctuation)

# load the provided stopwords
df_stop = pd.read_json('Datasets/Federaliststop.json')

# join provided stopwords with the default NLTK English stopwords
stopwords = set(df_stop['Term']).union(set(sw.words('english')))


#Methods
#-----------------------------------------------------------------------------------------


def lemmatize(token, tag):
    """
    Lemmatizers are pre-trained dictionaries that maps words and their semantic values into their lemma form. 

    Parameters
    ----------
    token:list[string], list of words 
    
    """
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

    return lemmatizer.lemmatize(token, tag)


def cab_tokenizer(document):
    """
    This function takes a document string, split it into tokens and preprocess them. Read the comments to understand the function of each part.

    Parameters
    ----------
    document:string, a document string for lemmitization
    """
    # initialize token list
    tokens = []
    
    # split the document into sentences
    for sent in sent_tokenize(document):
        # split the document into tokens and then create part of speech tag for each token
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If stopword, ignore token and continue
            if token in stopwords:
                continue

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # Lemmatize the token and add back to the tokens list
            lemma = lemmatize(token, tag)
            tokens.append(lemma)
    
    return tokens

def get_messages_similarity(messages,tfidf_vec):
    """
    This method gets the tweets of a user and calculates the average similarity of user's tweets

    Parameters
    ----------
    messages:list[string], list of user's tweets
    tfidf_vec: a model of TfidfVectorizer to calculat teh tf_idf parameter of each word


    """
    number_of_messages=len(messages)
    result=0
    if(number_of_messages>1):
        # tf idf vectoriser
        messages_tf_idf = tfidf_vec.fit_transform(messages)
        for i in range(number_of_messages):
            for j in range(i+1,number_of_messages):
                result += cosine_similarity(messages_tf_idf[i],messages_tf_idf[j])
    
    return result/number_of_messages

def get_avg_number_of_mentions(messages):
    """
    This method gets the tweets of a user and calculate the average number of '@' charchter in user's tweets

    Parameters
    ----------
    messages:list[string], list of user's tweets

    """
    number_of_messages=len(messages)
    result=0
    for i in range(number_of_messages):
        result+=messages[i].count('@')
    return result/number_of_messages

def create_database(spam_users_file_path,spam_tweets_file_path,nonspam_users_file_path,nonspam_tweets_file_path):
    """
    This method gets the paths of four honeypot datasets and creat a dataset (FinalDB.csv) based on seletced features
    
    Parameters
    ----------
    spam_users_file_path: spam_users.csv file path
    spam_tweets_file_path: spam_tweets.csv file path
    nonspam_users_file_path: nonspam_users.csv file path
    nonspam_tweets_file_path: nonspam_tweets.csv file path

    """

    #read database and store in 4 different variables
    spam_users=pd.read_csv(spam_users_file_path)
    spam_tweets=pd.read_csv(spam_tweets_file_path)
    nonspam_users=pd.read_csv(nonspam_users_file_path)
    nonspam_tweets=pd.read_csv(nonspam_tweets_file_path)

    # merge the spammers and legitimate users
    users = pd.concat([spam_users, nonspam_users])
    #merge the spammers'tweets and legitimate users' tweets
    tweets=pd.concat([spam_tweets, nonspam_tweets])


    tfidf_vec = TfidfVectorizer(tokenizer=cab_tokenizer)

    #create a data dataframe with columns equal to selected features
    dataframe= pd.DataFrame(columns=['followers_count', 'friends_count', '(followers_count/friends_count)',  'avg_of_numberOfHashtags_c','avg_of_numberOfMentions','avg_of_annotationMethod', 'message_similarirty', 'avg_of_maliciousMark', 'retweet_count','spam_or_nonspam'])
    
    #calculate the value of features for users
    for index, user in tqdm(users.iterrows()):

        #get the userID
        user_id=user['user']
        # Behavior Features:
        followers_count=user['followers_count'] 
        friends_count=user['friends_count'] 
        if(friends_count>0 and followers_count>0):
            followers_count_per_friends_count=format(followers_count/friends_count, '.2f')
        else:
            followers_count_per_friends_count=0
        spam_or_nonspam=user['spam_or_nonspam']

        # Content Features:
        #get the twwets of the user based on userID
        user_tweets=tweets.loc[tweets['user'] == user_id]
        if(len(user_tweets>0)):

            #calculate avergae number of hashtags in user's tweets
            numberOfHashtags_c=list(user_tweets['numberOfHashtags_c'])
            if(len(numberOfHashtags_c)>0):
                avg_of_numberOfHashtags_c=sum(numberOfHashtags_c)/len(numberOfHashtags_c)
                avg_of_numberOfHashtags_c=format(avg_of_numberOfHashtags_c, '.2f') 
            else:
                avg_of_numberOfHashtags_c=0

            #calculate avergae number of annotation Method in user's tweets
            annotationMethod=list(user_tweets['annotationMethod'])
            if(len(annotationMethod)>0):
                avg_of_annotationMethod=sum(annotationMethod)/len(annotationMethod)
                avg_of_annotationMethod=format(avg_of_annotationMethod, '.2f') 
            else:
                avg_of_annotationMethod=0

            #calculate avergae number of tweets' similarity of user
            messages=list(user_tweets['text'])
            message_similarirty=get_messages_similarity(messages,tfidf_vec)
            if(message_similarirty!=0):
                message_similarirty=format(message_similarirty[0][0], '.2f') 

            #calculate avergae number of malicious mark in user's tweets
            maliciousMark=sum(list(user_tweets['maliciousMark']))
            
            #calculate avergae number of retweet count in user's tweets
            retweet_count=list(user_tweets['retweet_count'])
            if(len(retweet_count)>0):
                avg_retweet_count=sum(retweet_count)/len(retweet_count)
            else:
                avg_retweet_count=0
            
            #calculate avergae number of mentions by @ in user's tweets
            avg_number_of_mentions=get_avg_number_of_mentions(messages)

        #if user has no tweets then:
        else:
            numberOfHashtags_c=0
            avg_of_numberOfHashtags_c=0
            message_similarirty=0
            maliciousMark=0
            avg_retweet_count=0
            avg_number_of_mentions=0
        
        #append a row to the dataframe which has values equal to calulcated features for selected user
        dataframe=dataframe.append({
                'followers_count':followers_count, 
                'friends_count':friends_count, 
                '(followers_count/friends_count)':followers_count_per_friends_count, 
                'avg_of_numberOfHashtags_c':avg_of_numberOfHashtags_c, 
                'avg_of_numberOfMentions':avg_number_of_mentions,
                'annotationMethod':avg_of_annotationMethod,
                'message_similarirty':message_similarirty, 
                'avg_of_maliciousMark':maliciousMark, 
                'retweet_count':avg_retweet_count,
                'spam_or_nonspam':spam_or_nonspam
                },ignore_index=True)

    #create FinalDB.csv file and save the created dataframe into that
    dataframe.to_csv("FinalDB.csv") 

def get_sample(number,x,y):
    idx=[]
    count=0
    for i in range(len(y)):
        if (count==number/2):
            break
        if(y[i]==0):
            idx.append(i)
            count+=1
    count=0
    for i in range(len(y)):
        if (count==number/2):
            break
        if(y[i]==1):
            idx.append(i)
            count+=1
    x = x.as_matrix()[idx]
    y=y.as_matrix()[idx]
    return x,y

def classification_report_perforemance(model,x_train, x_test,y_train,y_test):
    print("Train accuracy:", svm_model.score(x_train, y_train))
    print("Test accuracy:", svm_model.score(x_test, y_test))
    y_pred = svm_model.predict(x_test)
    print(classification_report(y_test, y_pred))

def get_dense_part(reachability,x,y):

    #get the denser part of clusters
    non_inf_reachability=reachability[reachability<max(reachability)]
    median_reachability=statistics.median(non_inf_reachability)
    index_list=[]
    for i in range(len(clust.reachability_)):
        if(reachability[i]<=median_reachability):
            index_list.append(i)
    x=np.nan_to_num(x[index_list])
    y=np.nan_to_num(y[index_list])
    return x,y

if __name__ == "__main__":



    #FinalDb.csv is genreated datafarame based on features selected from original datasets 
    try:
        FinalDb_df=pd.read_csv("FinalDB.csv")
    except IOError:
        #if FinalDb.csv is not generated, the file paths of four honeypot dataset should be given to the create_database function to generate the FinalDb.csv dataset
        spam_users_file_path='honeypot/spam_users.csv'
        spam_tweets_file_path='honeypot/spam_tweets.csv'
        nonspam_users_file_path='honeypot/nonspam_users.csv'
        nonspam_tweets_file_path='honeypot/nonspam_tweets.csv'
        create_database(spam_users_file_path,spam_tweets_file_path,nonspam_users_file_path,nonspam_tweets_file_path)
        print ("FinalDB.csv file is created successfully")
        FinalDb_df=pd.read_csv("FinalDB.csv")

    
    # divide the finalDb dataset to target and features values
    # y: target value
    # x: features value
    y = FinalDb_df['spam_or_nonspam']
    x = FinalDb_df.drop(['spam_or_nonspam',], axis=1)

    #get the 20000 portion of FinalDb.csv to implemet the hybrid model on 
    x,y=get_sample(20000,x,y)

    #Standardize features by removing the mean and scaling to unit variance
    x= StandardScaler().fit_transform(x)

    # calculate the cross validation of hybrid model
    scores = []

    # create cross validation object by 10-fold validation and random satate 42
    # n_splits: Number of folds
    cross_validation = KFold(n_splits=10, random_state=42, shuffle=False)

    #begin the evaluation of cross validation 
    for train_index, test_index in cross_validation.split(x):

        #split data to train and test data
        x_train=x[train_index]
        x_test=x[test_index]
        y_train=y[train_index]
        y_test=y[test_index]
        
        #Create OPTICS model
        #eps : The maximum distance between two samples for them to be considered as in the same neighborhood.
        #min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        #cluster_method: The extraction method used to extract clusters using the calculated reachability and ordering
        clust = OPTICS(min_samples=20,cluster_method='dbscan',eps=0.6)
        # Run the fit for OPTICS model
        clust.fit(x_train)

        #get the labeles, reachability distnace and spaces genearted by OPTICS model 
        space = np.arange(len(x_train))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]

        
        #get the denser part of clusters genrated by OPTICS model
        X_train_by_optics,Y_train_by_optics=get_dense_part(clust.reachability_,x_train,clust.labels_)

        #Create SVM model
        svm_model = SVC()
        # Run the fit for SVM model on denser part of clusters
        svm_model.fit(X_train_by_optics,Y_train_by_optics) 

        #get the performance of SVM on table
        classification_report_perforemance(svm_model,X_train_by_optics,x_test,Y_train_by_optics,y_test)
        
        #get the performance of SVM by cross validation
        scores.append(svm_model.score(x_test, y_test))

    #print the cross validation of hybrid model    
    print(np.mean(scores))