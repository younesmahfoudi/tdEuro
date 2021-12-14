from datetime import datetime
import pandas as pd 
import random
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from fastapi import FastAPI
from time import strftime

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

csv_file = 'EuroMillions_numbers.csv'  
df = pd.read_csv(csv_file, sep=';')
df = pd.DataFrame(df,columns=['Date','N1','N2','N3','N4','N5','E1','E2','Winner','Gain'])
df['Winner'] = df['Winner'].map('1'.format)



#Genere un tirage aléatoire, sans doublon et diférent de @param(combi)
def random_Combi(combi):
    n_list = random.sample(range(1,50), 5)
    e_list = random.sample(range(1,12), 2)
    list = n_list + e_list
    if (set(list) == set(combi)):
        random_Combi(combi)
    else:
        return list

#créer une une liste composé d'un @param(date), un @param(combi) et 2x 0 au format de la dataFrame   
def new_El(date, combi): 
    new_Combi = random_Combi(combi)
    new_el = [(date, new_Combi[0], new_Combi[1], new_Combi[2], new_Combi[3],new_Combi[4],new_Combi[5], new_Combi[6], '0')]
    return new_el

#creer une nouvelle dataFrame avec une date et une combinaison
def new_Df(date,combi):
    new_el = new_El(date,combi)
    new_df = pd.DataFrame(new_el, columns=['Date','N1','N2','N3','N4','N5','E1','E2','Winner'])
    return new_df

def new_ElWC(date, combi): 
    new_el = [(date, combi[0], combi[1], combi[2], combi[3],combi[4],combi[5], combi[6])]
    return new_el

#creer une nouvelle dataFrame avec une combi 
def new_DfWC(combi):
    date = datetime.today().strftime('%Y-%m-%d')
    new_el = new_ElWC(date_Converter(date),combi)
    new_df = pd.DataFrame(new_el, columns=['Date','N1','N2','N3','N4','N5','E1','E2'])
    return new_df

#extraire une combinaison d'un dataFrame
def read_Combi(df):
    return [df['N1'],df['N2'],df['N3'],df['N4'],df['N5'],df['E1'],df['E2']]


def date_Converter(date):
    d = datetime.strptime(date,'%Y-%m-%d')
    timestamp = datetime.timestamp(d)
    return timestamp

def tabToDf(predict):
    res = []
    for i in range(len(predict)):
        res.append(predict[i][0])
    
    df_res = pd.DataFrame(res,columns=['Predict'])
    return df_res

df['Date'] = df['Date'].apply(date_Converter)
del df['Gain']

#ajouter 4 combinaisons fausses
for row in df.index:
    for x in range(4):
        new_df = new_Df(df['Date'][row],read_Combi(df.iloc[4,:]))
        df = df.append(new_df, ignore_index=True)

#trie
df = df.sort_values(by=['Date'])

first_per = int((80*len(df))/100)

#split
X_train = df.iloc[:first_per,:]
X_test = df.iloc[first_per:,:]

y_train = X_train['Winner']
y_test = X_test['Winner']
del X_train['Winner']
del X_test['Winner']

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_predict = clf.predict_proba(X_test)

y_predict = tabToDf(y_predict)

y_predict = pd.concat([X_test,y_predict], axis=1, join='inner')

y_predict = y_predict.sort_values(by=['Predict'])

target_proba = y_predict[y_predict['Predict']>0.8]

test = target_proba.sample(n=1)

print(test)

class Combi(BaseModel):
    N1: int
    N2: int
    N3: int
    N4: int
    N5: int
    E1: int
    E2: int

class Probability(BaseModel):
    proba: float


@app.post("/api/predict")
async def predict_Combi(combi: Combi):
    res = clf.predict_proba(new_DfWC([combi.N1,combi.N2,combi.N3,combi.N4,combi.N5,combi.E1,combi.E2]))[0][0]
    return res

@app.get("/api/predict")
async def combi_Predict():
    res = target_proba.sample(n=1).to_json(orient = 'columns')
    return res

if __name__ == "__main__":

    test = clf.predict_proba(new_DfWC([7,12,18,23,32,4,12]))

    print(test)
    

    








    