#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install wordcloud')


# In[4]:


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from wordcloud import WordCloud #Word visualization

import re
import nltk
nltk.download('punkt')

from nltk import word_tokenize
nltk.download('stopwords')
print("modul loaded")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score,ConfusionMatrixDisplay


# In[5]:


#Validation dataset
val_data=pd.read_csv("../a_Week04_Lecture04_a/twitter_validation.csv", header=None)
#Full dataset for Train-Test
train_data=pd.read_csv("../a_Week04_Lecture04_a/twitter_training.csv", header=None)


# In[6]:


train_data.columns=['id','information','type','text']
train_data.head()


# In[7]:


val_data.columns=['id','information','type','text']
val_data.head()


# In[8]:


train_data.isna().sum()


# In[9]:


val_data.isna().sum()


# In[10]:


train_data.dropna(inplace=True)


# In[11]:


count=train_data['type'].value_counts().reset_index()
count.columns = ['type', 'count']
count


# In[12]:


fig = px.bar(count, x='type', y='count', title='Type Counts',color='type', color_discrete_sequence=px.colors.sequential.Plasma)
fig.update_layout(
    xaxis_title='Type', 
    yaxis_title='Count', 
    title_font_size=20,
    template='ggplot2' 
)
fig.show()


# In[13]:


#Text transformation
train_data["lower"]=train_data.text.str.lower() #lowercase
train_data["lower"]=[str(data) for data in train_data.lower] #converting all to string
train_data["lower"]=train_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex
val_data["lower"]=val_data.text.str.lower() #lowercase
val_data["lower"]=[str(data) for data in val_data.lower] #converting all to string
val_data["lower"]=val_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex


# In[14]:


train_data.tail()


# In[15]:


word_cloud_text = ''.join(train_data[train_data["type"]=="Positive"].lower)
#Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=80,                # Reduce the max font size to make the text more readable
    max_words=100,                   # Maximum number of words in the cloud
    background_color="white",       
    colormap='plasma',              
    scale=10,                     
    width=800, height=800,                      
    contour_width=1,              
).generate(word_cloud_text)
#Figure properties
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[16]:


word_cloud_text = ''.join(train_data[train_data["type"]=="Negative"].lower)
#Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="white",
    colormap='cividis',
    scale=10,
    width=800,
    height=800,
    contour_color='black',   
).generate(word_cloud_text)
#Figure properties
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[17]:


word_cloud_text = ''.join(train_data[train_data["type"]=="Irrelevant"].lower)
#Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="white",
    colormap='viridis',
    scale=10,
    width=800,
    height=800
).generate(word_cloud_text)
#Figure properties
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[18]:


word_cloud_text = ''.join(train_data[train_data["type"]=="Neutral"].lower)
#Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="white",
    scale=10,
    width=800,
    height=800,
    colormap='cividis',
    contour_color='purple'  
).generate(word_cloud_text)
#Figure properties
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[20]:


plot1=train_data.groupby(by=["information","type"]).count().reset_index()
plot1[['information',  'id']]


# In[21]:


fig = px.bar(
    plot1,
    x="information",
    y="id",
    color="type",
    labels={"information": "Brand", "id": "ID Number of tweets"},  # Customizing labels
    title="Distribution of tweets per Brand and Type"  # Title of the plot
)

# Update layout for better styling
fig.update_layout(
    xaxis_title="Brand",
    yaxis_title="ID Number of tweets",
    xaxis_tickangle=-90,  # Rotate x-axis labels
    title_font_size=20,  # Title font size
    xaxis_title_font_size=16,  # X-axis title font size
    yaxis_title_font_size=16,  # Y-axis title font size
    template="plotly_white"  # Use a white background for better contrast
)

# Show the figure
fig.show()


# In[22]:


#Text splitting
tokens_text = [word_tokenize(str(word)) for word in train_data.lower]
#Unique word counter
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))


# In[23]:


tokens_text
tokens_counter 


# ## TfidfVectorizer

# In[24]:


tf=TfidfVectorizer(max_features=5000)
x=tf.fit_transform(train_data['lower'])
y=train_data['type']


# In[25]:


x.shape


# In[26]:


y.shape


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)


# In[28]:


models = {
    'Naive Bayes': MultinomialNB(),
    'Decision tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[33]:


disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='coolwarm', normalize='true')
disp.figure_.set_facecolor('white') 
disp.ax_.grid(False)
disp.ax_.set_xlabel('Predicted Label', fontsize=14)
disp.ax_.set_ylabel('True Label', fontsize=14)
disp.ax_.set_title('Confusion Matrix', fontsize=16)
disp.ax_.tick_params(axis='both', which='major', labelsize=12)
plt.show()

