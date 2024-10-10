# **Symbolic Computing**
```
import sympy
sympy.init_printing()   #works in some non-interactive environment

whos

x= sympy.Symbol("x",real=True)
y=sympy.Symbol("y")

a,b=sympy.symbols("a,b")

sympy.sqrt(x**2)

sympy.Rational(11,3)

g=sympy.Function('g')(a,b)

g.free_symbols

sympy.Lambda(a,sympy.factorial(a))

expr=x**3-4*x**2-8*x+4*x*y
expr.args[1].args

expr.simplify()
expr.expand()
sympy.factor(expr)
expr.collect(x)             #only for the first power

expr=y / (y * x + y)+ 1 / (1+x)
sympy.together(expr)
sympy.ratsimp(expr)

[expr.subs(x,num).evalf(2) for num in range(0,3)]

sympy.diff(g,a,4)
sympy.integrate(f,(x,a,b))

from sympy.plotting import plot

plot(f,g,xlim,ylim,xlabel,ylabel,legend)
```
------------------------------------------------------------------------------------------------------------


### List
```
s.replace('','',1)  # for replacing in series in python
''.join()
''.split()
.append()      # append at the end
.insert[0,1]   # append at the specific location

.pop()         #remove items, the last by default
```
#### or
```
.remove()

.index(value, start, end)

.sort(s,reverse)
```
#### or
```
sorted(s)
```
### Dictionary
```
del(dict[])     # remove key/value pair from dictionary
dict1.upadate(dict2)   # merge dictionaries
dict.items()/.keys()/.values()
max(dict1,key=dict1.get)   # .get a function getting specific value 
np.argmax
```
------------------------------------------------------------------------------------------------------------
### Function
```
assert condition,"Optional error message"
isinstance(a,int)

def function(func,arg,*list)          # arg=1,tuple=(2,45),function are first-class objects
function(pow,1,2,45)           

def function(**kwargs:dict)->dict       # for dictionary
function(Sean=67,Ben=72,Anne=66)
function._annotations_           # inspect type hints  

if :
    return              # base case
return l[0]+l[1:]       # Recursive function for cumsum
```
#### or 
```
first,*rest=L
return first if not rest else first+mysum(rest)
```
#### or
```
return reduce(function,iterable,initializer)

yield i                 # like print
 
global variable
nonlocal variable

changer(x,l[:])        # lists are more mutable, so we need to make a shallow copy 

tuple1=[(function,args),(function,args)]
for func,arg in tuple1:
    x(y)                 # batch functions iterator
```
------------------------------------------------------------------------------------------------------------

### Class
```
class function:
    b=''                        # class-level data
    def __init__(self,nam=default):
        self.name=nam
        
    def __eq__(self,other):
        return self.name==other.name
    def __ne__(self,other):
        return self.name!=other.name
    def __ge__(self,other):
        return self.name>=other.name
    def __le__(self,other):
        return self.name<=other.name
    def __gt__(self,other):
        return self.name>other.name
    def __lt__(self,other):
        return self.name<other.name
    
    def __repr__(self):        # for developers
    def __str__(self):         # for users   
        
class function2(function):
    def __init__(self,nam,ag):
        
        super().__init__(nam)
        #or
        funtion.__init__(self,nam)
        
        self.age=ag
a=function()

class CustomError(Exception):    # inherit from built-in Exception base class
    pass
try:
    if 
    raise CustomError
except CustomError:              # raising exceptions manually
    print('')
finally:                         # executes after trying
    print
```

------------------------------------------------------------------------------------------------------------
### File
```
with open('text.txt','w')as f:      
    f.write()
with open('text.txt','r')as f:
    contents=f.read()
    all_lines=f.readlines()
```
#### or 
```
zip(iterable1, iterable2, ...)
map(func, iterable)
filter(func, iterable)
iter()
f.readline()
f.__next__()
    
with open("my_first_file.txt", "a") as f:  
json.loads(str)             # string outputs list or dict
json.dumps([])              #list or dict outputs string
```
### contextmanager 
```
@contextlib.contextmanager
def cont():
    print()
    try:
        yield
    except:
        yield 
    print()
with cont() as outp: 
    print(outp)                  # define the contextmanager first and then input values 
```
### Decorator
```
def default(condition):
    def decorator(func):
        def dec1():
            return func()
        return dec1
    return decorator
@decorator
def function():
    return
```     
### Numpy (good at multidimensions and vectors)
```
np.arange()
np.array()
np.shape()
.sum()
.mean()
.median()
np.random.rand()
np.ones((,))
np.absolute()
```
------------------------------------------------------------------------------------------------------------
```
expr=x**3-2*x+5          # Newton's Method
initial_guess =[1,3]
for guess in intial_guess:
     sol = sympy.nsolve(expr, x, guess)
    if 1 <= sol <= 2 or sympy.exp(1) <= sol <= 4:
        solutions.append(sol)
```



# **Dataset**
```
import pandas as pd
import numpy as np

dat=pd.read_csv('',parse_dates=[''])
```
### Statistical information
```
dat.info()
dat.head()
dat.shape
dat.size     # total number of elements 
dat.dtypes
dat.describe()  
dat.columns  # .index  if there exists index, then it will work
dat.sample(frac,n)

dat.value_counts()
dat.groupby()
dat.loc[]       # for index / .iloc[[],[]] for position 
dat.sort_values('',ascending)

import plotly.graph_objects as go    # drawing table
go.Figure(data=[go.Table(header=dict(values,fill_color,align),cells=dict(values,fill_color,align))])
```
------------------------------------------------------------------------------------------------------------
### Indexing & Related Manipulating
```
dat.set_index()        # set columns to indexes
dat.reindex()          # conform to new indexes or columns 
dat.reset_index(drop,level)       # back index column to value, add default integer index
dat.reset_index()[names]
dat.sort_index(axis,ascending) 
pd.concat([],axis=1)          # stacking data based on same indexes: 1 is row
pd.merge(dat1,dat2,how,on)    # depending on which datasets' indexes you want
dat[~dat.isin()]         #diagnose the joins
pd.melt(dat,id_vars,var_name,value_name)  # melt from wide format to long format
pd.pivot(dat,index,columns,values)     # melt from long format to wide format
for i in educ.columns: educ[i + '_log'] = np.log(educ[i])   
```
### Manipulating
```
dat.query('a>3 and b<5')    # more readable way
dat.agg()       # eg:agg({'A': 'sum', 'B': ['mean', 'std']})
dat.assign()     # creating new mutiple columns
list(map(lambda x:x[:2],dat.date))  # batch processing only for series
```
#### or
```
.apply(lambda if else for,axis)   # more flexible for dataframe
```
#### or for lists
```
[lambda for x in if ]
```
#### or 
```
np.where(condition,x,y)
```


### Quantile Cuts
```
dat.quantile(q=[])
pd.qcut(data,q,labels)
```
### NA and Duplicates and Outliers  - 1.drop na 2.categorical strings to values 3.np.log
```
dat.drop()
dat.dropna()
dat[dat.isna().any(axis=1)]
dat.drop_duplicates()
np.log()
dat.unique()[::-1]    # delete duplicates and reverse the array
```
#### or
```
set(s)
dat.clip(lower,upper,axis,inplace)   # a given range to data
```

### Categorical variables
```
np.where(condition,x,y)   # convert categories to numbers
pd.Categorical(dat,categories,ordered=True)
dat.cat.set_categories()   #update categories
dat.cat.reorder_categories
dat.cat.add_categories()      # .cat is necessary for accessing categorical variables
dat.cat.remove_categories(removals)
dat.cat.rename_categories(new_categories={})
dat.cat.rename_categories({})
```
### String variables
```
dat.str.strip()   # Strip out whitespaces
dat.str.title()
dat.replace({})   # for collapsing or editing in pandas
```
### Dates and Times
```
from datetime import date
from datetime import timedelta
from datetime import datetime 
from datetime import timezone 

pd.to_datetime(date,format='ISO8601')   # making timestamps, same as parse_dates

date(2014.8.30).year/month/day/weekday()
(date1-date2).days/.total_seconds
date.today()+timedelta(days=30)

dtm(2024,8,30,15,14)
date.strftime('%Y %m %d %B %D %H %M %S')    # turning time into strings
dtm.strptime(dat,'%m/%d/%Y %H:%M:%S') #turning strings into corresponding time

PDT=tmz(timedelta(hours))
dtm(2021,5,30,tzinfo=PDT)   # only can fill integers
```
#### or 
```
dat.astimezone(PDT)      # more flexible
```


# **Plots**
```
dat.hist()
```
### Plt
```
import matplotlib.pyplot as plt

plt.figure(figsize=())

figure, axes = plt.subplots(2, 2)    # multiple subplots 
axes[0, 0].scatter(x , y, color , marker , linestyle, linewidth)    #scatter
axes[0, 1].bar(x, y)
ax3=axes[0,1].twinx()      # creates a secondary axis on the right
axes[1, 0].hist(x,bins,histtype,label)
axes[2, 0].boxplot(x,positions)

ax.set_title('')
ax.legend()
ax.set_xlabel('')
ax.tick_params('', colors)
ax.despine()

plt.show()
```
------------------------------------------------------------------------------------------------------------

### Seaborn

```
import seaborn as sns

sns.displot(data,x,kind='hist/ked/ecdf',rug,kde,kde_kws,stat,palette) # continuous
sns.catplot(data,x,y,hue,kind,order,legend_out)  # categorical
```
### lineplot
```
sns.lineplot(x,y)
```
### Scatter
```
sns.relplot(data,x,y,hue,col) # continuous * continuous
```
### Regression
```
sns.lmplot(data,x,y,hue,robust,ci,order)  # continuous * continuous, specifically regression
```
#### or
```
sns.regplot(x,y,data)
```
#### or logistic regression
```
sns.regplot(x,y,data,logistic = True)
```
#### Residual
```
sns.residplot(x,y,data,lowess)

sns.jointplot(data,x,y,hue,kind)    # continuous * continuous,with distribution plots 

tab=pd.crosstab(x,y,values,aggfunction,normalize = 'index') # Discrete * Discrete
tab.cumsum(axis).stack().reset_index(name)     # axis=0 column, axis=1 row

sns.set_palette()  
```
------------------------------------------------------------------------------------------------------------
### Plotnine-ggplot
```
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap

(
    ggplot(dat,aes(x,y,color))
    +geom_point()          # various kinds
    +stat_smooth(method='lm')   # statistical layer
    +facet_wrap(factor)            # one-variable subplots
)
```
------------------------------------------------------------------------------------------------------------
### Cartopy
```
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

lon = np.linspace(-180, 180, 360)
lat = np.linspace(-90, 90, 180)
lon, lat = np.meshgrid(lon, lat)
data = np.sin(np.radians(lat)) * np.cos(np.radians(lon)) # adding longitude and latitude data
plt.figure(figsize=(10, 5))
ax2 = plt.axes(projection=ccrs.PlateCarree())   # choose a projection(Mercator,Orthographic)
ax2.set_global() 
contour = ax2.contourf(lon, lat, data,
                       transform=ccrs.PlateCarree(), cmap='RdBu')   # fill the contour plot(optional)
ax2.coastlines() 
plt.show()
```
------------------------------------------------------------------------------------------------------------
### Plotly
```
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

px.histogram(dat,x,color,nbins,marginal,histnorm, title,labels={'title':{'text':'My title'}})
px.bar(dat,x,y,color,barmode,hover_data)
px.box(dat,x,y,color,notched,points)
px.scatter(dat,x,y,color,title,hover_data,marginal_X,marginal_y,trendline,log_x,log_y)
px.scatter_matrix(dat,dimensions)

corm=educ.corr(method,numeric_only)       # correlation data, method= 'kendall' and 'spearman'，‘pearson’
px.imshow(corm,color_continuous_scale,zmin,zmax)    # heatmap,image plots

g.update_layout(title={'text': 'Updated Title'},width,height)
g.updata_traces(diagnol_visible)     # figures on the diagnol line
g.update_xaxes(title_text='X Label')
g.update_yaxes(title_text='Y Label')
```
### Complete control over figures
```
f=go.Figure()
```
#### or
```
f=make_subplots(row,cols,subplot_titles)

f.add_trace(go.Bar(name,x,y,error_y))
f.add_trace(go.Histogram(x,nbinsx,name),row,col)  # using for to add more traces
f.add_trace(go.Scatter(x,y,mode,name))
```
### dropdown_buttons
```
dropdown_buttons=[dict(label,method,args=[{'visible': [True, False, False, False]},{'title': 'Education'}])]
f.update_layout(barmode,updatemenus=['type': "dropdown",'x': 1.3,'y': 0.5,'showactive': True,'active': 0,'buttons': dropdown_buttons])
```
### sliders
```
px.scatter(dat,x,y,color,size,animation_frame,animation_group,log_x,size_max,range_x,range_y)

fig['layout'].pop('updatemenus')  # for removing the interactive buttons
f.show()
```


# **Regression**

### Cross-Validation
```
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
```
#### 1. Validation set approach
```
X_train, X_test,y_train,y_test= train_test_split(X,y,test_size,random_state)
```
#### 2.LOOCV
```
cv=LeaveOneOut()
model=LinearRegression()
get_scorer_names()       # list of scoring
np.mean(np.absolute(scores))
```
```
for train_index,test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
#### 3.K-fold 
```
cv=KFold(n_splits,random_state,shuffle)
```
#### 4.K-nearest neighbors classifier
```
knn= KNeighborsClassifier(n_neighbors=10).fit(X,y)
```
##### search best K
```
bigK = list(range(1, 100))
for i in bigK:
    cv = KFold(n_splits = 10, random_state = 1234, shuffle = True)
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=corss_val_score(knn,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
    errmea.append(1-scores.mean())
print(f'best k is {str(bigK[errmea.index(min(errmea))])}')
sns.lineplot(x=bigK,y=errma)    
plt.scatter(bigK[errmea.index(min(errmea))], min(errmea), marker='X', color = 'red')
plt.text(x,y,f'(x,y)',color,fontsize)
```
#### 4.Subset selection
##### Best subset selection (linear regression-RSS; logistic regression-deviance)
```
def powerset(s):
    sets=[[]]
    for i in s:
        newsets=[]
        for k in sets:
            newsets.append(k+[i])
        sets+=newsets
    return sets
def best_subset_selection(est,X,y,cvn=5):
    modsAll=powerSet(X.columns)
    mods={}
    for i in modsAll:
        if i==[]:
            mods[0]={'model':'Null',
                    'RSS':np.sum((y-np.mean(y)**2))}
        else:
            rss=np.sum((y-est.fit(X[i],y).predict(X[i]))**2)
            if len(i) not in mods:
                mods[len(i)]={'model':i,'RSS':rss}
            else:
                if mods[len(i)]['RSS']>rss:
                    mods[len(i)]={'model':i,'RSS':rss}
    bestmod=mods.pop(0)
    bestmod.pop('RSS')
    bestmod['Rsquared']=0
    for i in mods.values():
        score=cross_val_score(est,X[i['model']],y,cv=cvn,scoring='r2').mean()
        if bestmod['Rsquared']<score:
            bestmod['model']=i['model']
            bestmod['Rsquared']=score
    return(bestmod)
```
##### Stepwise selection: forward selection
```
def forward_subset_selection(est,X,y,cvn=5):
    variables=list(X.columns)
    mods={}
    mods[0]={'model':[],'RSS':np.sum(y-np.mean(y)**2)}
```    
        
##### Stepwise selection: backward selection
##### Cp,AIC,BIC,Adjusted R2

### Statsmodel
```
import statsmodels.api as sm
kx1=sm.add_constant(X_train,prepend=False)        # fits a regression model
kx2=sm.add_constant(X_test)
model=sm.OLS(y_train,kx1).fit()
y_pred=model.predict(kx2)
model.summary()
```
### statsmodel more readable way(but difficult for data splitting)
```
from statsmodels.formula.api import ols

model=ols('y~x+z',data).fit()
model.params  # get co-efficients
model.fittedvalues # get fitted values
model.resid # get residuals
```
### Scikit-learn
```
from sklearn.linear_model import LinearRegression
reg= LinearRegression().fit(X_train,y_train)
y_pre=reg.predict(X_test)
```
### Diagnostics
```
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import plot_partregress_grid, influence_plot

model.summary()
```
### R-squared  proportion of explained variables
```
model.rsquared
```
### MSE
```np.sum((y_pre-y_test)**2)
mean_squared_error(X, y)
```
### MSEresid
```
model.mse_resid
```
### RSE
```
np.sqrt(mse)
```
### Heteroscedasticity
```
sns.residplot(x,y,data,lowess)
```
### Outlier
```
model.get_influence().summary_frame()
```
### Studentized_resid  or np.abs()  -2~2
```
sns.regplot(x = 'fittedvalues', y = 'student_resid', x_jitter = 0.1, y_jitter = 0.1, data = summary_info, lowess = True)
```
### Leverage - hat_diag >(k+1)/n
```
sns.scatterplot(x = 'hat_diag', y = 'student_resid', data = summary_info)
```
### leverage x cook's-d >1
```
sns.scatterplot(x = 'hat_diag', y = 'cooks_d', data = summary_info)
```
### Multicollinearity
```
sns.pairplot(data[[]])
variance_inflation_factor(data.values,i) for i in range(data.shape[1])  #shape[0] gives the number of rows. shape[1] gives the number of columns.
```
### Model Improvement
```
anova_lm(model, model3)
```
### Model plots
```
plot_partregress_grid(model)
influence_plot(model)
```
### CV MSE
```
scores=cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=cv)
np.mean(np.absolute(scores))
```
### CV ErR
```
scores=corss_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
1-scores.mean()
```
------------------------------------------------------------------------------------------------------------

## Logistic Regression

### Statsmodel
```
from statsmodels.formula.api import logit

logit('y~x', data = chile_clean).fit()
```
# Scikit-learn
from sklearn.linear_model import LogisticRegression

# create dummies
data['result']=np.where()
pd.get_dummies(data,prefix,drop_first=True)
dummies.astype(int)
pd.concat()
X=dat[[]]
y=dat[]

# check polynomial 
for p in list(range(1,4)):
    if p==1:
        X=pd.DataFrame({
            'age_1': data['age']
        })
    else:
        X['age_'+str(p)]=X['age_1']**p

model=LogisticRegression(solver= 'newton-cg')
model.fit(X,y)

model.intercept_      # or np.exp() 
model.coef_


------------------------------------------------------------------------------------------------------------


# Generative models of classification
#1. Linear Discriminant Analysis- different from logistic regression, needs Gaussian Distribution in X and similar covariance in y
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

X,y=dat1,dat2
model=LinearDiscriminantAnalysis()
model.fit(X, y)
DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict"/'predict_proba',
                                             alpha=0.5, cmap=plt.cm.coolwarm)
fig.ax_.scatter(x = chile_clean['logincome'], y = chile_clean['age'],c=chile_clean['vote'],alpha=0.5,cmap=plt.cm.coolwarm)

#2.Quadratic Discriminant Analysis
QuadraticDiscriminantAnalysis()
model.fit(X, y)

#3.Naive Bayes
GaussianNB()
model.fit(X, y)

y_pre=model.predict(X_test)

# Confusion Matrix
confusion_matrix(y, y_pre)
# Classification Report
classification_report(y, y_pre)










from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
