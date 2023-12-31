import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from IPython.display import display

np.random.seed(1)

iris = sklearn.datasets.load_iris()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

sklearn.metrics.accuracy_score(labels_test, rf.predict(test))

print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
    
i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)

html_output = exp.as_html()

with open('output.html', 'w') as file:
    file.write(html_output)

""" res = exp.show_in_notebook(show_table=True, show_all=False)

with open('output.html', 'w') as file:
    file.write(res.data) """