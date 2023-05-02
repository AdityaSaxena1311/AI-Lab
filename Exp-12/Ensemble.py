import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
data = pd.DataFrame({
    'feature1': [0, 1, 0, 1, 0, 1, 1, 0],
    'feature2': [0, 1, 1, 1, 0, 1, 0, 0],
    'target': [0, 1, 1, 1, 0, 1, 0, 0]
})
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
model3 = KNeighborsClassifier()
#ensemble model
ensemble_model = VotingClassifier(estimators=[('dt', model1), ('lr', model2), ('knn', model3)], voting='hard')
ensemble_model.fit(X_train, y_train)
accuracy = ensemble_model.score(X_test, y_test)
print(f"Ensemble model accuracy: {accuracy}")
