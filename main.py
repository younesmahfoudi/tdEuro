from datetime import datetime
import pandas as pd 
import random
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

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
    new_el = [(date, new_Combi[0], new_Combi[1], new_Combi[2], new_Combi[3],new_Combi[4],new_Combi[5], new_Combi[6], '0', '0')]
    return new_el

#creer une nouvelle dataFrame avec une date et une combinaison
def new_Df(date,combi):
    new_el = new_El(date,combi)
    new_df = pd.DataFrame(new_el, columns=['Date','N1','N2','N3','N4','N5','E1','E2','Winner','Gain'])
    return new_df

#extraire une combinaison d'un dataFrame
def read_Combi(df):
    return [df['N1'],df['N2'],df['N3'],df['N4'],df['N5'],df['E1'],df['E2']]

def date_Converter(date):
    d = datetime.strptime(date,'%Y-%m-%d')
    timestamp = datetime.timestamp(d)
    return timestamp
    

if __name__ == "__main__":

    df['Date'] = df['Date'].apply(date_Converter)

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

    # modele_rf = RandomForestClassifier(
    #     n_estimators=100,
    #     criterion='gini',
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     min_weight_fraction_leaf=0.0,
    #     max_features='auto',
    #     max_leaf_nodes=None,
    #     min_impurity_decrease=0.0,
    #     bootstrap=True,
    #     oob_score=False,
    #     n_jobs=None,
    #     random_state=None,
    #     verbose=0,
    #     warm_start=False,
    #     class_weight=None,
    #     ccp_alpha=0.0,
    #     max_samples=None,)
    

    # X=df.drop(['N1'],axis=1)
    # y=df['N1']
    # x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=123)

    # x_train, x_test, y_train, y_test  = train_test_split(df1, 
    #                                                     df2, 
    #                                                     test_size=0.20, 
    #                                                     random_state=42)

    # modele_rf.fit(x_train, y_train)

    y_train = X_train['Winner']
    y_test = X_test['Winner']
    del X_train['Winner']
    del X_test['Winner']

    print(y_train)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)


    y_predict = clf.predict_proba(X_test)
    print(y_predict)


    








    