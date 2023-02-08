import bnlearn as bn
import bnlearn.inference
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import networkx as nx



def confusion_matrix(y_teste, y_pred):


    conf_matrix = metrics.confusion_matrix(y_teste, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predições', fontsize=18)
    plt.ylabel('Real', fontsize=18)
    plt.title('Matriz de Confusão', fontsize=18)
    plt.show()
def inferencia(teste, modelo):
    predicoes_corretas = []
    predicoes = []
    count = 0
    y_teste = []
    tp = []
    fv = []
    total = []
    for idx, row in teste.iterrows():

            if row.name == 1797 or row.name == 43783 or row.name == 1912 or row.name == 16241:
                pass
            else:



                resultado = bnlearn.inference.fit(modelo, variables=["HeartDisease"],
                                           evidence={"BMI": row["BMI"], "Smoking": row["Smoking"],
                                                     "AlcoholDrinking": row["AlcoholDrinking"], "Stroke": row["Stroke"],
                                                     "PhysicalHealth": row["PhysicalHealth"],
                                                     "MentalHealth": row["MentalHealth"],
                                                     "DiffWalking": row["DiffWalking"], "Sex": row["Sex"],
                                                     "AgeCategory": row["AgeCategory"],
                                                     "Diabetic": row["Diabetic"], "Race": row["Race"],
                                                     "PhysicalActivity": row["PhysicalActivity"],
                                                     "GenHealth": row["GenHealth"], "SleepTime": row["SleepTime"],
                                                     "Asthma": row["Asthma"], "KidneyDisease": row["KidneyDisease"],
                                                     "SkinCancer": row["SkinCancer"]}, verbose=0)

                y_teste.append(row['HeartDisease'])

                if (resultado.df['p'][1] >= 0.60 and row['HeartDisease'] == 1):
                    predicoes_corretas.append(1)
                    tp.append(1)
                elif (resultado.df['p'][1] < 0.60 and row['HeartDisease'] == 0):
                    predicoes_corretas.append(0)
                elif (resultado.df['p'][1] < 0.60 and row['HeartDisease'] == 1):
                    fv.append(0)




    acuracia = len(predicoes_corretas) / len(y_teste)
    sensibilidade = len(tp) / (len(tp) + len(fv))
    # confusion_matrix(y_teste, predicoes)

    return acuracia, sensibilidade


def strafiyClass(base):
    split = StratifiedShuffleSplit(test_size=0.15)
    for x, y in split.split(base, base["HeartDisease"]):
        base_treinamento = base.iloc[x]
        base_teste = base.iloc[y]

    return base_treinamento, base_teste








if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    base = pd.read_csv("baseEstratificada.csv")
    #Tratamento da base de dados


    base2 = pd.read_csv('heart_2020_cleaned.csv')
    X = base.drop(columns="HeartDisease")
    y = base['HeartDisease']


    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size = 0.25, random_state = 0)


    X_train["HeartDisease"] = y_train
    X_test["HeartDisease"] = y_test



    # cat = [col for col in base.columns if base[col].dtypes == 'object']
    # import seaborn as sns
    # plt.figure(figsize=(16, 8))
    # sns.catplot(y = 'HeartDisease', x='DiffWalking', data=base, kind='bar')
    # plt.show()

    model = bn.structure_learning.fit(base, methodtype='hc', scoretype='k2')
    dag_update = bnlearn.parameter_learning.fit(model, X_train)
    # bn.plot(dag_update, interactive=True )

    grafo = nx.Graph()
    edges = model['model_edges']
    nodes = ["HeartDisease",    "BMI",   "Smoking",  "AlcoholDrinking",
             "Stroke",  "PhysicalHealth",
             "MentalHealth",  "DiffWalking",    "Sex",  "AgeCategory",
             "Race",  "Diabetic",  "PhysicalActivity",  "GenHealth",
             "SleepTime",  "Asthma",  "KidneyDisease",  "SkinCancer"]

    for node in nodes:
        grafo.add_node(node)

    for edge in edges:
        grafo.add_edge(edge[0], edge[1])


    nx.draw_networkx(grafo, with_labels=True)
    plt.show()
    # homens_teste = X_test[X_test['Sex'] == "Male"]
    # mulheres_teste = X_test[X_test['Sex'] == "Female"]
    #
    # brancos_teste = X_test[X_test['Race'] == "White"]
    # pretos_teste = X_test[X_test['Race'] == "Black"]
    #

    #
    #
    # acuracia, sensibilidade = inferencia(X_test, dag_update)
    # print('------------------------------')
    # print(f"Acurácia total: {acuracia}")
    # print(f"Sensibilidade total: {sensibilidade}")
    # #
    # acuracia, sensibilidade = inferencia(homens_teste, dag_update)
    # print('------------------------------')
    # print(f"Acurácia Homens: {acuracia}")
    # print(f"Sensibilidade Homens: {sensibilidade}")
    #
    # acuracia, sensibilidade = inferencia(mulheres_teste, dag_update)
    # print('------------------------------')
    # print(f"Acurácia Mulheres: {acuracia}")
    # print(f"Sensibilidade Mulheres: {sensibilidade}")
    #
    # acuracia, sensibilidade = inferencia(brancos_teste, dag_update)
    # print('------------------------------')
    # print(f"Acurácia Brancos: {acuracia}")
    # print(f"Sensibilidade Brancos: {sensibilidade}")
    # #
    #
    # acuracia, sensibilidade = inferencia(pretos_teste, dag_update)
    # print('------------------------------')
    # print(f"Acurácia Pretos: {acuracia}")
    # print(f"Sensibilidade Pretos: {sensibilidade}")
    # #


    # base_treinamento, base_teste = strafiyClass(mulheres_pretas)
    # model = bn.structure_learning.fit(base_treinamento, scoretype='k2')
    # dag_update = bnlearn.parameter_learning.fit(model, base_treinamento)
    # acuracia = inferencia(base_teste, dag_update)
    # print(f"Acurácia mulheres pretas: {acuracia}")
    # bn.plot(model, interactive=True)
    #
    # base_treinamento, base_teste = strafiyClass(mulheres_brancas)
    # model = bn.structure_learning.fit(base_treinamento, scoretype='k2')
    # dag_update = bnlearn.parameter_learning.fit(model, base_treinamento)
    # acuracia = inferencia(base_teste, dag_update)
    # print(f"Acurácia mulheres brancas: {acuracia}")
    # bn.plot(model, interactive=True)
    #






