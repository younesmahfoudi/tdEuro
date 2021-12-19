from datetime import datetime
import pandas as pd 
import random
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from fastapi import FastAPI, HTTPException
import numpy as np

from time import strftime

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

csv_file = 'EuroMillions_numbers.csv'  
df = pd.read_csv(csv_file, sep=';')
df = pd.DataFrame(df,columns=['Date','N1','N2','N3','N4','N5','E1','E2','Winner','Gain'])
df['Winner'] = df['Winner'].map('1'.format)



def random_Combi(combi):
    '''
    Genere un tirage aléatoire, sans doublon et différent de @param(combi)

            Parameters:
                    combi : 
            Returns:
                    list : 
    '''
    n_list = random.sample(range(1,50), 5)
    e_list = random.sample(range(1,12), 2)
    list = n_list + e_list
    if (set(list) == set(combi)):
        random_Combi(combi)
    else:
        return list

def new_El(date, combi): 
    '''
    créer une une liste composé d'un @param(date), un @param(combi) et 2x 0 au format de la dataFrame   

            Parameters:
                    date : 
                    combi
            Returns:
                    new_el : 
    '''
    new_Combi = random_Combi(combi)
    new_el = [(date, new_Combi[0], new_Combi[1], new_Combi[2], new_Combi[3],new_Combi[4],new_Combi[5], new_Combi[6], '0')]
    return new_el


def new_Df(date,combi):
    '''
    creer une nouvelle dataFrame avec une date et une combinaison

            Parameters:
                    date : 
                    combi :
            Returns:
                    new_df : 
    '''
    new_el = new_El(date,combi)
    new_df = pd.DataFrame(new_el, columns=['Date','N1','N2','N3','N4','N5','E1','E2','Winner'])
    return new_df

def new_ElWC(date, combi): 
    '''
    creer une nouvel élément avec une date et une combinaison

            Parameters:
                    date : 
                    combi :
            Returns:
                    new_el : 
    '''
    new_el = [(date, combi[0], combi[1], combi[2], combi[3],combi[4],combi[5], combi[6])]
    return new_el


def new_DfWC(combi):
    '''
    creer une nouvelle dataFrame avec une combi 

            Parameters:
                    combi :
            Returns:
                    new_df : Dataframe avec une date et une combinaison d'élements
    '''
    date = datetime.today().strftime('%Y-%m-%d')
    new_el = new_ElWC(date_Converter(date),combi)
    new_df = pd.DataFrame(new_el, columns=['Date','N1','N2','N3','N4','N5','E1','E2'])
    return new_df


def read_Combi(df):
    '''
    extraire une combinaison d'un dataFrame

            Parameters:
                    df :
            Returns:

    '''
    return [df['N1'],df['N2'],df['N3'],df['N4'],df['N5'],df['E1'],df['E2']]


def date_Converter(date):
    '''
    Conversion date en seconde

            Parameters:
                    date : date
            Returns:
                    timestamp : 

    '''
    d = datetime.strptime(date,'%Y-%m-%d')
    timestamp = datetime.timestamp(d)
    return timestamp

def tabToDf(predict):
    '''
    Conversion tableau to Dataframe

            Parameters:
                    predict : 
            Returns:
                    df_res : DataFrame 

    '''
    res = []
    for i in range(len(predict)):
        res.append(predict[i][0])
    
    df_res = pd.DataFrame(res,columns=['Predict'])
    return df_res

def predictConverter(predict):
    '''

            Parameters:
                    predict : 
            Returns:
                    res : entier 

    '''
    if (predict >= 0.6):
        res = 1
    else:
         res = 0
    return res

def compareTwoDF(df1, df2, attribute):
    '''
    Comparer deux dataframe
            Parameters:
                    df1 : Dataframe
                    df2 : Dataframe 
            Returns:
    '''
    res = 0
    for row in df1.index:
        if df1[attribute][row] == df2[attribute][row]:
            res += 1
    return res/df1.index
    
def verifValeurCombinaison(combinaison):
    '''
    Vérifie les nombres saisies pour jouer à l'Euromillions

            Parameters:
                    combinaison : Combinaison
            Return:
                    bool : boolean ( valeurs saisies valides ou non ) 
    '''
    bool = True
    for key, value in combinaison:
        if(value>0):
            if((key=="E1") or (key=="E2")):
                if((value>12)):
                    bool = False
            else :
                if((value>50)):
                    bool = False 
        else: 
            bool = False

    return bool          


# Conversion des dates en secondes 
df['Date'] = df['Date'].apply(date_Converter)
# Suppresion de la colonne Gain
del df['Gain']

#ajouter 4 combinaisons fausses
for row in df.index:
    for x in range(4):
        new_df = new_Df(df['Date'][row],read_Combi(df.iloc[4,:]))
        df = df.append(new_df, ignore_index=True)

#trie
df = df.sort_values(by=['Date'])

# 80 %
first_per = int((80*len(df))/100)

#split des DF
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

#print(y_predict)

y_predict_metric = y_predict

# Application de la métrique 
y_predict_metric['Predict'] = y_predict_metric['Predict'].apply(predictConverter)

y_predict = pd.concat([X_test,y_predict], axis=1, join='inner')

y_predict = y_predict.sort_values(by=['Predict'])

# Application de la métrique 
print(y_test.value_counts())
# y_predict_metric['Winner'] = y_test
win_metric = abs(y_predict_metric['Predict'].value_counts()[1] - y_test.value_counts()[1])
metric = (win_metric/len(y_test))*100

print(y_predict_metric['Predict'].value_counts())
print(len(y_test))
print(metric)
# print(y_predict_metric)


# Ensemble de combinaison avec une probabilité de 0.8 ( valeur que nous avons choisi)
target_proba = y_predict[y_predict['Predict']>0.8]

# Class combinaison
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
    if(verifValeurCombinaison(combi)):
        res = clf.predict_proba(new_DfWC([combi.N1,combi.N2,combi.N3,combi.N4,combi.N5,combi.E1,combi.E2]))[0][0]
        return res
    else : 
        raise HTTPException(status_code=404, detail="Nombres saisies invalides : les 5 premiers numéros doivent être compris entre 1 et 50 inclus et les 2 derniers numéros étoiles de 1 à 12.")


@app.get("/api/predict")
async def combi_Predict():
    res = target_proba.sample(n=1)
    # Dataframe to list
    res = res.astype(int).to_numpy().tolist()
    # Suppression date
    del res[0][0]
    #List list to list
    flattened = [val for sublist in res for val in sublist]
    res = flattened
    res = json.dumps(res)
    return "Combinaison: " + res


# à implémenter
@app.get("/api/model")
async def get_Infos_Model():
    metriques = metric
    algo = clf.__class__.__name__
    param = "?"
    return {"Metriques de performance (pourcentage d'érreur)" : metriques, "Nom de l'algo" : algo, "Paramètres d'entraînement" : param }

# à implémenter
@app.put("api/model")
async def add_Data(combi: Combi):
    if(verifValeurCombinaison(combi)):
        new_df = new_DfWC([combi.N1,combi.N2,combi.N3,combi.N4,combi.N5,combi.E1,combi.E2])
        df.append(new_df, ignore_index=True)
        print(df)
        return {"Donnée ajoutée"}
    else : 
        raise HTTPException(status_code=404, detail="Nombres saisies invalides : les 5 premiers numéros doivent être compris entre 1 et 50 inclus et les 2 derniers numéros étoiles de 1 à 12.")
    

# à implémenter
@app.post("api/model")
async def retrain_Model():
    #split des DF
    X_train = df.iloc[:first_per,:]
    X_test = df.iloc[first_per:,:]
    y_train = X_train['Winner']
    y_test = X_test['Winner']
    del X_train['Winner']
    del X_test['Winner']
    clf.fit(X_train, y_train)
    return {"Modèle réentrainé"}



if __name__ == "__main__":

    test = clf.predict_proba(new_DfWC([7,12,18,23,32,4,12]))
    #print(y_predict)
    #print(compareTwoDF(y_predict_metric, y_test, 'Predict'))
    #print(y_predict_metric)
    #print(test)
    

    






