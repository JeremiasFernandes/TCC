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
    predicoes = []
    predicoes_matrix = []
    for idx, row in teste.iterrows():
        result = bnlearn.inference.fit(modelo, variables=["HeartDisease"],
                                       evidence={"BMI": row["BMI"], "Smoking": row["Smoking"],
                                                 "AlcoholDrinking": row["AlcoholDrinking"], "Stroke": row["Stroke"],
                                                 "PhysicalHealth": row["PhysicalHealth"],
                                                 "MentalHealth": row["MentalHealth"],
                                                 "DiffWalking": row["DiffWalking"], "Sex": row["Sex"],
                                                 "AgeCategory": row["AgeCategory"],
                                                  "Diabetic": row["Diabetic"],
                                                 "PhysicalActivity": row["PhysicalActivity"],
                                                 "GenHealth": row["GenHealth"], "SleepTime": row["SleepTime"],
                                                 "Asthma": row["Asthma"], "KidneyDisease": row["KidneyDisease"],
                                                 "SkinCancer": row["SkinCancer"]}, verbose=0)

        if result.df['p'][1] >= 0.80:
            predicoes_matrix.append(1)
        if result.df['p'][1] < 0.80:
            predicoes_matrix.append(0)

        if (result.df['p'][1] >= 0.80 and row['HeartDisease'] == 1):
            predicoes.append(1)
        elif (result.df['p'][1] >= 0.80 and row['HeartDisease'] == 0):
            predicoes.append(0)
        elif (result.df['p'][1] < 0.80 and row['HeartDisease'] == 1):
            predicoes.append(0)
        elif (result.df['p'][1] < 0.80 and row['HeartDisease'] == 0):
            predicoes.append(1)

    cont = 0
    for i in predicoes:
        if i == 1:
            cont += 1

    acuracia = cont / len(predicoes)


    confusion_matrix(teste['HeartDisease'], predicoes_matrix)
    return acuracia


def strafiyClass(base):
    split = StratifiedShuffleSplit(test_size=0.15)
    for x, y in split.split(base, base["HeartDisease"]):
        base_treinamento = base.iloc[x]
        base_teste = base.iloc[y]

    return base_treinamento, base_teste








if __name__ == '__main__':

    base = pd.read_csv("heart_2020_cleaned.csv")
    #Tratamento da base de dados

    #Descretizando o atributo "Diabetic".
    diabetes = {'Yes': 1, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3}
    base['Diabetic'] = base['Diabetic'].apply(lambda x: diabetes[x])
    base['Diabetic'] = base['Diabetic'].astype('int')
    #O atributo idade é rotulado como um intervalo entre idades, mas este formato é um complicador para o aprendizado
    #Neste trabalho, foi escolhido a média entre os intervalos para subtituir os dados.
    idades = {'55-59':57, '80 or older':80, '65-69':67, '75-79':77, '40-44':42, '70-74':72,
           '60-64':62, '50-54':52, '45-49':47, '18-24':21, '35-39':37, '30-34':32, '25-29':27}
    base['AgeCategory'] = base['AgeCategory'].apply(lambda x: idades[x])
    base['AgeCategory'] = base['AgeCategory'].astype('float')
    base['BMI'] = base['BMI'].astype('int')



    # base.loc[base['SleepTime'] < 7] = 6
    # base.loc[base['SleepTime'] > 10] = 10
    # base.loc[base['SleepTime'] > 7] = 8

    aux = LabelEncoder()
    aux.fit(base['HeartDisease'])
    base['HeartDisease'] = aux.transform(base['HeartDisease'])

    aux.fit(base['Smoking'])
    base['Smoking'] = aux.transform(base['Smoking'])

    aux.fit(base['AlcoholDrinking'])
    base['AlcoholDrinking'] = aux.transform(base['AlcoholDrinking'])

    aux.fit(base['Stroke'])
    base['Stroke'] = aux.transform(base['Stroke'])

    aux.fit(base['DiffWalking'])
    base['DiffWalking'] = aux.transform(base['DiffWalking'])


    aux.fit(base['PhysicalActivity'])
    base['PhysicalActivity'] = aux.transform(base['PhysicalActivity'])

    aux.fit(base['GenHealth'])
    base['GenHealth'] = aux.transform(base['GenHealth'])

    aux.fit(base['Asthma'])
    base['Asthma'] = aux.transform(base['Asthma'])

    aux.fit(base['KidneyDisease'])
    base['KidneyDisease'] = aux.transform(base['KidneyDisease'])

    aux.fit(base['SkinCancer'])
    base['SkinCancer'] = aux.transform(base['SkinCancer'])

    aux.fit(base['Sex'])
    base['Sex'] = aux.transform(base['Sex'])

    aux.fit(base['Race'])
    base['Race'] = aux.transform(base['Race'])

    aux.fit(base['Diabetic'])
    base['Diabetic'] = aux.transform(base['Diabetic'])

    aux.fit(base['GenHealth'])
    base['GenHealth'] = aux.transform(base['GenHealth'])


    X = base.drop(columns="HeartDisease")
    y = base['HeartDisease']


    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size = 0.20, random_state = 3, stratify=y)

    #
    # X_train['HeartDisease'] = y_train
    # X_test['HeartDisease'] = y_test
    # model = bn.structure_learning.fit(X_train, methodtype='tan', class_node='HeartDisease')
    # dag_update = bnlearn.parameter_learning.fit(model, X_train)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print("Chegou aqui")


    result = accuracy_score(y_test, predictions)
    confusion_matrix(y_test, predictions)
    print(result)

    # brancos_treinamento = base_treinamento[base_treinamento['Race'] == "White"]
    # pretos_treinamento = base_treinamento[base_treinamento['Race'] == "Black"]
    # mulheres_pretas_treinamento = pretos_treinamento[pretos_treinamento['Sex'] == 'Female']
    # mulheres_brancas_treinamento = brancos_treinamento[brancos_treinamento['Sex'] == 'Female']
    # # homens_pretos = pretos[pretos['Sex'] == 'Male']
    # # homens_brancos = brancos[brancos['Sex'] == 'Male']

    # brancos_teste = base_teste[base_teste['Race'] == "White"]
    # pretos_teste = base_teste[base_teste['Race'] == "Black"]
    # mulheres_pretas_teste = pretos_teste[pretos_teste['Sex'] == 'Female']
    # mulheres_brancas_teste = brancos_teste[brancos_teste['Sex'] == 'Female']
    # homens_pretos_teste = pretos[pretos['Sex'] == 'Male']
    # homens_brancos = brancos[brancos['Sex'] == 'Male']

    # brancos_teste = brancos_teste.head(len(mulheres_pretas_teste))
    # pretos_teste = pretos_teste.head(len(mulheres_pretas_teste))
    # mulheres_brancas_teste = mulheres_brancas_teste.head(len(mulheres_pretas_teste))
    #
    #
    #
    #
    # acuracia = inferencia(X_test, dag_update)
    # print(f"Acurácia total: {acuracia}")
    # bn.plot(model, interactive=True)

    # acuracia = inferencia(brancos_teste, dag_update)
    # print(f"Acurácia brancos: {acuracia}")
    # #
    #
    #
    # acuracia = inferencia(pretos_teste, dag_update)
    # print(f"Acurácia pretos: {acuracia}")
    # #
    # #
    # acuracia = inferencia(mulheres_pretas_teste, dag_update)
    # print(f"Acurácia mulheres pretas: {acuracia}")
    # #
    #
    # acuracia = inferencia(mulheres_brancas_teste, dag_update)
    # print(f"Acurácia mulheres brancas: {acuracia}")
    #

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






