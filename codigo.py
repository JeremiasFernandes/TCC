import bnlearn as bn
import bnlearn.inference
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit











base = pd.read_csv("heart_2020_cleaned.csv")


#stratify da base
split = StratifiedShuffleSplit(test_size=0.1)
for x, y in split.split(base, base["HeartDisease"]):
    df_x = base.iloc[x]
    df_y = base.iloc[y]


#
model = bn.structure_learning.fit(df_x)
dag_update = bnlearn.parameter_learning.fit(model,df_x)
#


for idx, row in df_y.iterrows():
    result = bnlearn.inference.fit(dag_update, variables=["HeartDisease"], evidence={"BMI":row["BMI"],"Smoking":row["Smoking"],
                                                                             "AlcoholDrinking":row["AlcoholDrinking"],"Stroke":row["Stroke"],
                                                                             "PhysicalHealth":row["PhysicalHealth"],"MentalHealth":row["MentalHealth"],
                                                                             "PhysicalHealth":row["PhysicalHealth"],"MentalHealth":row["MentalHealth"],
                                                                             "DiffWalking":row["DiffWalking"],"Sex":row["Sex"],"AgeCategory":row["AgeCategory"],
                                                                             "Race":row["Race"],"Diabetic":row["Diabetic"],"PhysicalActivity":row["PhysicalActivity"],
                                                                             "GenHealth":row["GenHealth"], "SleepTime":row["SleepTime"],"Asthma":row["Asthma"],"KidneyDisease":row["KidneyDisease"],
                                                                             "SkinCancer":row["SkinCancer"]})
    print(type(result))
 