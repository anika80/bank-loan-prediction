import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

bankdata = pd.read_csv("train_data.csv")
bankdata.drop(['Loan_ID'], inplace = True, axis = 1)
# transforming the values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
bankdata['Gender'] = le.fit_transform(bankdata['Gender'])
bankdata['Married'] = le.fit_transform(bankdata['Married'])
bankdata['Education'] = le.fit_transform(bankdata['Education'])
bankdata['Self_Employed'] = le.fit_transform(bankdata['Self_Employed'])
bankdata['Property_Area'] = le.fit_transform(bankdata['Property_Area'])
bankdata['Loan_Status'] = le.fit_transform(bankdata['Loan_Status'])
#Dropping the null values
bankdata = bankdata.dropna(axis = 0)
bankdata['Dependents'].replace('3+','4',inplace=True)
bankdata['Dependents']=bankdata['Dependents'].astype(int)

#ML_Model
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=14, shuffle = True, random_state=3)

x = bankdata.drop('Loan_Status', axis=1)
y = bankdata['Loan_Status']

#Random Forest
for train_index, test_index in skf.split(x, y):
    x_train, x_test, y_train, y_test = x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_jobs=-1, bootstrap= False, criterion= 'gini', max_depth= 10, max_features= 2, min_samples_leaf= 2, n_estimators= 200)
rfc.fit(x_train, y_train) #Training the model
y_pred_rfc = rfc.predict(x_test)


pickle.dump(rfc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
