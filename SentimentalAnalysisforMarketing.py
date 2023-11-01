import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
df = pd.read_csv(r"C:\Users\matha\Downloads\patrick\Tweets.csv")
df.head()
null = df.isnull().sum()/len(df)
null
df.drop(['airline_sentiment_gold','negativereason_gold','tweet_coord'],axis=1,inplace=True)
print(df['airline_sentiment'].value_counts())
sns.countplot(data=df,x='airline_sentiment')
plt.show()
df.airline.value_counts()
def plot_sentiment(Airline):
    df1 = df[df['airline']==Airline]
    count=df1['airline_sentiment'].value_counts().reset_index().rename(columns={'index':Airline,'airline_sentiment':'count'})
    sns.barplot(data=count,x=Airline,y='count')
plt.figure(figsize=(15,7))
plt.subplot(2,3,1)
plot_sentiment('United')
plt.subplot(2,3,2)
plot_sentiment('US Airways')
plt.subplot(2,3,3)
plot_sentiment('American')
plt.subplot(2,3,4)
plot_sentiment('Southwest')
plt.subplot(2,3,5)
plot_sentiment('Delta')
plt.subplot(2,3,6)
plot_sentiment('Virgin America')
plt.tight_layout()
plt.show()
df['negativereason'].value_counts()
def plot_reason(Airline):
    df2 = df[df['airline']==Airline]
    count=df2['negativereason'].value_counts().reset_index().rename(columns={'index':'negativereason','negativereason':'count'})
    sns.barplot(data=count,x='negativereason',y='count')
    plt.title('Count of Reasons for '+Airline)
    plt.xticks(rotation=90)
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plot_reason('United')
plt.subplot(2,3,2)
plot_reason('US Airways')
plt.subplot(2,3,3)
plot_reason('American')
plt.subplot(2,3,4)
plot_reason('Southwest')
plt.subplot(2,3,5)
plot_reason('Delta')
plt.subplot(2,3,6)
plot_reason('Virgin America')
plt.tight_layout()
plt.show()
from wordcloud import WordCloud,STOPWORDS

df3 = df[df['airline_sentiment']=='negative']
words = ' '.join(df3['text'])
cleaned_word = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word!='RT'])

wordcloud = WordCloud(background_color='black',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
import nltk
import re
from nltk.corpus import stopwords
def tweet_len(tweet):
    letters_only = re.sub('^[a-zA-Z]',' ',tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_word = [w for w in words if w not in stops]
    return (len(meaningful_word))
df['sentiment'] = df['airline_sentiment'].apply(lambda x:0 if x=='negative' else 1)
df['clean_tweet'] = df['text'].apply(lambda x:tweet_to_words(x))
df['tweet_length'] = df['text'].apply(lambda x:tweet_len(x))
train, test = train_test_split(df,test_size=0.2,random_state=42)
train_clean_tweet = []
for tweet in train['clean_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet = []
for tweet in test['clean_tweet']:
    test_clean_tweet.append(tweet)
    from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer='word')
train_features = v.fit_transform(train_clean_tweet)
test_features = v.transform(test_clean_tweet)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))   
    result = pd.DataFrame({'Models':Model})
result['Accuracy'] = Accuracy
result = result.sort_values(by='Accuracy',ascending=False)
result