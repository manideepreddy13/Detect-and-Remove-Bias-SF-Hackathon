import pandas as pd
from collections import Counter
from sklearn.cluster import kmeans_plusplus
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
#import lightgbm as lgb
import xgboost as xgb
import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from lime import submodular_pick
pd.set_option('display.max_columns', None)


def stat_analyse(data_df,cat_var,stat_var,data_levels = [0,5,10,25,50,70,75,80,90,95,99,100]):
    data_created_dict = dict()
    for cat in data_df[cat_var].unique():
        cur_data = data_df.loc[data_df[cat_var]==cat][stat_var]
        data_created_dict[cat] = []
        for level in data_levels:
            data_created_dict[cat].append(cur_data.quantile(level/100))
    return (data_created_dict)


compas_data = pd.read_csv('compas-scores-raw.csv')
compas_data = compas_data.loc[compas_data['DisplayText']=='Risk of Violence'].loc[compas_data['AssessmentType']=='New']
compas_data.set_index('AssessmentID',inplace=True)

compas_data[compas_data['Person_ID']==12368]

#print(compas_data)

compas_data['Scale_ID'].value_counts()

#print(compas_data)

AnalysisResult = stat_analyse(compas_data,'Language','DecileScore',[0,10,20,30,40,50,60,70,80,90,100])
exdf = pd.DataFrame.from_dict(AnalysisResult, orient='index',columns=[0,10,20,30,40,50,60,70,80,90,100])

#print(exdf)

AnalysisResult = stat_analyse(compas_data,'Ethnic_Code_Text','DecileScore',[0,10,20,30,40,50,60,70,80,90,100])
exdf = pd.DataFrame.from_dict(AnalysisResult, orient='index',columns=[0,10,20,30,40,50,60,70,80,90,100])

#print(exdf)

final_compas_data = compas_data[['Agency_Text', 'Sex_Code_Text', 'Ethnic_Code_Text', 'DateOfBirth','Screening_Date', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'RecSupervisionLevel', 'ScoreText']]

final_compas_data['Screening_Date'] = final_compas_data['Screening_Date'].str.split(' ',n=1)
#final_compass_data =
rp = final_compas_data.Screening_Date.apply(pd.Series)
final_compas_data['Screening_Date'] = rp[0]
final_compas_data['Screening_Date'] = pd.to_datetime(final_compas_data['Screening_Date'].str[:-2] + '20' + final_compas_data['Screening_Date'].str[-2:])
final_compas_data['DateOfBirth'] = pd.to_datetime(final_compas_data['DateOfBirth'].str[:-2] + '19' + final_compas_data['DateOfBirth'].str[-2:])
#final_compas_data['Screening_Date'] = final_compas_data['Screening_Date'].to_list()
final_compas_data['Screening_Date'] = pd.to_datetime(final_compas_data['Screening_Date'])
final_compas_data['DateOfBirth'] = pd.to_datetime(final_compas_data['DateOfBirth'])
final_compas_data['DOB'] = final_compas_data['Screening_Date']-final_compas_data['DateOfBirth']
final_compas_data['DOB'] = final_compas_data['DOB'].dt.days
temp_data = final_compas_data[['Ethnic_Code_Text', 'Sex_Code_Text','Agency_Text', 'LegalStatus', 'CustodyStatus', 'MaritalStatus','RecSupervisionLevel']]
one_hot_encoded_data = pd.get_dummies(temp_data, columns = ['Ethnic_Code_Text', 'Sex_Code_Text','Agency_Text', 'LegalStatus', 'CustodyStatus', 'MaritalStatus','RecSupervisionLevel'])
one_hot_encoded_data['DOB'] = final_compas_data['DOB']
one_hot_encoded_data['ScoreText'] = final_compas_data['ScoreText']

final_compas_data = one_hot_encoded_data

final_compas_data.dropna(inplace=True)
final_compas_data['ScoreText'] = pd.Categorical(final_compas_data.ScoreText)
final_compas_data['ScoreText'] = final_compas_data.ScoreText.cat.codes


target = final_compas_data['ScoreText']
features = final_compas_data.drop(['ScoreText'], axis=1)
X = features.values
y = target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Random Forest Classifier
""" clf = RandomForestClassifier(n_estimators = 1000, random_state = 72)
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print(cm)
 """

y_train = to_categorical(y_train,3)
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu')) # input shape is (features,)
model.add(Dense(3, activation='softmax'))
model.summary()

# compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', # this is different instead of binary_crossentropy (for regular classification)
              metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   patience=10,
                                   restore_best_weights=True) # important - otherwise you just return the last weigths...

# now we just update our model fit call
history = model.fit(X_train,
                    y_train,
                    callbacks=[es],
                    epochs=8000, # you can set this to a big number!
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)

y_pred = model.predict(X_test)
y2 = y_pred
y_pred = [np.argmax(line) for line in y_pred]
print(y_pred)

from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=final_compas_data.columns, class_names=[0,1,2], discretize_continuous=True)
i = np.random.randint(0, y_test.shape[0])
exp = explainer.explain_instance(X_test[i], model.predict, num_features=40, top_labels=1)

lime_output = exp.as_html()

sp_obj = submodular_pick.SubmodularPick(explainer, X_train, model.predict, sample_size=30, num_features=10, num_exps_desired=15)

#print(sp_obj)

""" splime_output_elem = [exp.as_html(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]

splime_output = ''.join(splime_output_elem)

print(splime_output) """



splime_output_elem = []

for exp in sp_obj.sp_explanations:
    splime_output_elem.append(exp.as_html())
            
splime_output = ''.join(splime_output_elem)

with open("output.html","w") as fl:
    fl.write(splime_output)