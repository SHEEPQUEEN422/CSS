from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

print(os.getcwd())
anes = pd.read_csv("/home/hjunyi/b_ProblemSet_Week08_ds1.5/anes2016.csv")
anes[['age_cont','feel_rep_cand_cont']]
anes['vote_pres_2016'].unique()
anes.columns
anes.head()

#1.Fit a KNN model that predicts vote in the 2016 election based on the feeling thermometer (feel_rep_cand_cont) and the age (age_cont).
#Find the optimal K between 1 and 21, 2 by 2. Use 4-Fold cross-validation to search for the best K.
#Plot the optimal K
#Fit the model with the optimal K and use cross-validation, saving 35 percent of the data for testing set.
#Print the confusion matrix and the classification report.

X=anes[['age_cont','feel_rep_cand_cont']]
y=anes['vote_pres_2016']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=42)
k_values=range(1,22,2)
best_k=1
best_score=0
erromea=[]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=4,scoring='accuracy')
    mean_score=np.mean(scores)
    erromea.append(1-mean_score)
    if mean_score>best_score:
        best_score=mean_score
        best_k=k      
print(best_k,best_score,erromea)

sns.lineplot(x=k_values,y=erromea)    
plt.scatter(k_values[erromea.index(min(erromea))], min(erromea), marker='X', color = 'red')

knn_best=KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train,y_train)
cv_best_scores = cross_val_score(knn_best, X_train, y_train, cv=4, scoring='accuracy')
test_accuracy = knn_best.score(X_test, y_test)
print(test_accuracy,cv_best_scores)

y_pred=knn_best.predict(X_test)
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
class_report=classification_report(y_test,y_pred)
print(class_report)

#2. Fit a KNN model that predicts vote in the 2016 election using all variables-- find optimal K.
X=anes[['pay_attn_pol_cont', 'int_follow_campg', 'anything_like_dem_cand',
       'anything_like_rep_cand', 'approve_congr', 'things_right_track',
       'feel_dem_cand_cont', 'feel_rep_cand_cont', 'how_many_live_hh_cont',
       'better_1y_ago_cont', 'has_hinsur', 'favor_aca', 'afraid_dem_cand',
       'disgust_dem_cand', 'afraid_rep_cand', 'disgust_rep_cand',
       'lib_con_scale_cont', 'incgap_morethan_20y_ago', 'economy_improved',
       'unempl_improved', 'speaksmind_dem_cand', 'speaksmind_rep_cand',
       'soc_spend_favor_cont', 'def_spend_favor_cont', 'private_hi_favor_cont',
       'shoud_hard_buy_gun', 'favor_affirmaction', 'govt_benefit_all',
       'all_ingovt_corrup', 'election_makegovt_payattn',
       'global_warming_happen', 'favor_death_penalty',
       'econ_better_since_2008', 'relig_important', 'age_cont', 'married',
       'schooling_cont', 'latinx', 'white', 'black', 'both_parents_bornUS',
       'any_grandparent_foreign', 'rent_home', 'has_unexp_passap',
       'should_roughup_protestors', 'justified_useviolence',
       'consider_self_feminist', 'ppl_easily_offended_nowadays',
       'soc_media_learn_pres', 'satisfied_life']]
y=anes['vote_pres_2016']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=42)
k_values=range(1,22,2)
best_k=1
best_score=0
erromea=[]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=4,scoring='accuracy')
    mean_score=np.mean(scores)
    erromea.append(1-mean_score)
    if mean_score>best_score:
        best_score=mean_score
        best_k=k      
print(best_k,best_score,erromea)

sns.lineplot(x=k_values,y=erromea)    
plt.scatter(k_values[erromea.index(min(erromea))], min(erromea), marker='X', color = 'red')

#3.Fit a KNN model that predicts vote on Trump and Clinton in the 2016 election, using all variables--find optimal K.
TC_anes=anes[(anes['vote_pres_2016']=='Trump')|(anes['vote_pres_2016']=='Clinton')]
X=TC_anes[['pay_attn_pol_cont', 'int_follow_campg', 'anything_like_dem_cand',
       'anything_like_rep_cand', 'approve_congr', 'things_right_track',
       'feel_dem_cand_cont', 'feel_rep_cand_cont', 'how_many_live_hh_cont',
       'better_1y_ago_cont', 'has_hinsur', 'favor_aca', 'afraid_dem_cand',
       'disgust_dem_cand', 'afraid_rep_cand', 'disgust_rep_cand',
       'lib_con_scale_cont', 'incgap_morethan_20y_ago', 'economy_improved',
       'unempl_improved', 'speaksmind_dem_cand', 'speaksmind_rep_cand',
       'soc_spend_favor_cont', 'def_spend_favor_cont', 'private_hi_favor_cont',
       'shoud_hard_buy_gun', 'favor_affirmaction', 'govt_benefit_all',
       'all_ingovt_corrup', 'election_makegovt_payattn',
       'global_warming_happen', 'favor_death_penalty',
       'econ_better_since_2008', 'relig_important', 'age_cont', 'married',
       'schooling_cont', 'latinx', 'white', 'black', 'both_parents_bornUS',
       'any_grandparent_foreign', 'rent_home', 'has_unexp_passap',
       'should_roughup_protestors', 'justified_useviolence',
       'consider_self_feminist', 'ppl_easily_offended_nowadays',
       'soc_media_learn_pres', 'satisfied_life']]
y=TC_anes['vote_pres_2016']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=42)
k_values=range(1,22,2)
best_k=1
best_score=0
erromea=[]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=4,scoring='accuracy')
    mean_score=np.mean(scores)
    erromea.append(1-mean_score)
    if mean_score>best_score:
        best_score=mean_score
        best_k=k      
print(best_k,best_score,erromea)

#4.Try:Logistic Regression
TC_anes['vote_pres_2016']=np.where(TC_anes['vote_pres_2016']=='Clinton',0,1)
X=TC_anes[['pay_attn_pol_cont', 'int_follow_campg', 'anything_like_dem_cand',
       'anything_like_rep_cand', 'approve_congr', 'things_right_track',
       'feel_dem_cand_cont', 'feel_rep_cand_cont', 'how_many_live_hh_cont',
       'better_1y_ago_cont', 'has_hinsur', 'favor_aca', 'afraid_dem_cand',
       'disgust_dem_cand', 'afraid_rep_cand', 'disgust_rep_cand',
       'lib_con_scale_cont', 'incgap_morethan_20y_ago', 'economy_improved',
       'unempl_improved', 'speaksmind_dem_cand', 'speaksmind_rep_cand',
       'soc_spend_favor_cont', 'def_spend_favor_cont', 'private_hi_favor_cont',
       'shoud_hard_buy_gun', 'favor_affirmaction', 'govt_benefit_all',
       'all_ingovt_corrup', 'election_makegovt_payattn',
       'global_warming_happen', 'favor_death_penalty',
       'econ_better_since_2008', 'relig_important', 'age_cont', 'married',
       'schooling_cont', 'latinx', 'white', 'black', 'both_parents_bornUS',
       'any_grandparent_foreign', 'rent_home', 'has_unexp_passap',
       'should_roughup_protestors', 'justified_useviolence',
       'consider_self_feminist', 'ppl_easily_offended_nowadays',
       'soc_media_learn_pres', 'satisfied_life']]
y=TC_anes['vote_pres_2016']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print('confusion matrix for logistic regression:',cm,'accuracy score for logistic regression:',accuracy)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Trump', 'Trump'], yticklabels=['Not Trump', 'Trump'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

#5.LDA
lda_model=LinearDiscriminantAnalysis()
lda_model.fit(X_train,y_train)
y_pred=lda_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('accuracy for linear discriminant analysis is:',accuracy)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix for linear discriminant analysis is:',cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Trump', 'Trump'], yticklabels=['Not Trump', 'Trump'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
report=classification_report(y_test,y_pred)
print('classificaiton report for linear discriminant analysis is:',report)

#6.Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
nb_model=GaussianNB()
nb_model.fit(X_train,y_train)
y_pred=nb_model.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
cm=confusion_matrix(y_test,predictions)
print('accuracy for naive bayes is:',accuracy)
print('confusion matrix for naive bayes is:',cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Trump', 'Trump'], yticklabels=['Not Trump', 'Trump'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
