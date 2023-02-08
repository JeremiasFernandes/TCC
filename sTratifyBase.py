import pandas as pd
from sklearn.preprocessing import LabelEncoder


base = pd.read_csv("heart_2020_cleaned.csv")
# Tratamento da base de dados

# Descretizando o atributo "Diabetic".
diabetes = {'Yes': 1, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3}
base['Diabetic'] = base['Diabetic'].apply(lambda x: diabetes[x])
base['Diabetic'] = base['Diabetic'].astype('int')
# O atributo idade é rotulado como um intervalo entre idades, mas este formato é um complicador para o aprendizado
# Neste trabalho, foi escolhido a média entre os intervalos para subtituir os dados.
idades = {'55-59': 57, '80 or older': 80, '65-69': 67, '75-79': 77, '40-44': 42, '70-74': 72,
          '60-64': 62, '50-54': 52, '45-49': 47, '18-24': 21, '35-39': 37, '30-34': 32, '25-29': 27}
base['AgeCategory'] = base['AgeCategory'].apply(lambda x: idades[x])
base['AgeCategory'] = base['AgeCategory'].astype('float')
base['BMI'] = base['BMI'].astype('int')

print(base)
# base.loc[base['SleepTime'] < 7] = 6
# base.loc[base['SleepTime'] > 10] = 10
# base.loc[base['SleepTime'] > 7] = 8

aux = LabelEncoder()

aux.fit(base['HeartDisease'])
base['HeartDisease'] = aux.transform(base['HeartDisease'])

# aux.fit(base['Smoking'])
# base['Smoking'] = aux.transform(base['Smoking'])
#
# aux.fit(base['AlcoholDrinking'])
# base['AlcoholDrinking'] = aux.transform(base['AlcoholDrinking'])
#
# aux.fit(base['Stroke'])
# base['Stroke'] = aux.transform(base['Stroke'])
#
# aux.fit(base['DiffWalking'])
# base['DiffWalking'] = aux.transform(base['DiffWalking'])
#
# aux.fit(base['PhysicalActivity'])
# base['PhysicalActivity'] = aux.transform(base['PhysicalActivity'])

# aux.fit(base['GenHealth'])
# base['GenHealth'] = aux.transform(base['GenHealth'])

# aux.fit(base['Asthma'])
# base['Asthma'] = aux.transform(base['Asthma'])
#
# aux.fit(base['KidneyDisease'])
# base['KidneyDisease'] = aux.transform(base['KidneyDisease'])
#
# aux.fit(base['SkinCancer'])
# base['SkinCancer'] = aux.transform(base['SkinCancer'])
#
# aux.fit(base['Sex'])
# base['Sex'] = aux.transform(base['Sex'])

# aux.fit(base['Race'])
# base['Race'] = aux.transform(base['Race'])

# aux.fit(base['Diabetic'])
# base['Diabetic'] = aux.transform(base['Diabetic'])

# aux.fit(base['GenHealth'])
# base['GenHealth'] = aux.transform(base['GenHealth'])

registrosPositivos = base[base['HeartDisease'] == 1]
registrosNegativos = base[base['HeartDisease'] == 0]
registrosNegativos = registrosNegativos.head(30000)

newdataFrame = pd.concat([registrosPositivos, registrosNegativos])


newdataFrame.to_csv('baseEstratificada.csv', index=False)

