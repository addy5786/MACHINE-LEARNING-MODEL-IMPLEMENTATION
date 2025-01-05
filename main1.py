import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
df=pd.read_csv("mail_data.csv")
df.head()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

df.isnull().sum()
df.shape
df.info()
df['Category']=df['Category'].map({'spam':0,'ham':1})
df.head()
X=df['Message']
Y=df['Category']
X
Y
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
X_train.shape
y_train.shape
X_test.shape
feature_extraction=TfidfVectorizer(min_df=1,stop_words="english",binary=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)
X_train
X_train_features
X_test_features
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train_features,y_train)
prediction_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train,prediction_train_data)
print("Accuracy on train data:" ,accuracy_train_data)
prediction_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test,prediction_test_data)
print("Accuracy on train data:",accuracy_test_data)
# Building predictive system
input_user_mail=["Thanks for your subscription to Ringtone UK your mobile will be charged Â£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"]
input_data_features = feature_extraction.transform(input_user_mail)
prediction = model.predict(input_data_features)
if prediction[0]==1:
    print("This is a Ham mail")
else:
    print("This is a Spam mail")
