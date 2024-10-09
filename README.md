# Basic
## Symbolic Computing
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


## Dataset
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
