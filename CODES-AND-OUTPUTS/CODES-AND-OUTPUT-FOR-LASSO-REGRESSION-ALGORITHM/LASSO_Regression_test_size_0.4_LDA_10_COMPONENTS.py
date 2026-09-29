import numpy as np
import pandas as pd

df1 = pd.read_csv('C:/Users/mithun/OneDrive/Documents/ml_PYTHON/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('C:/Users/mithun/OneDrive/Documents/ml_PYTHON/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('C:/Users/mithun/OneDrive/Documents/ml_PYTHON/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

#feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=10)
X_train = lda.fit_transform(X_train, y_train) # needs labels
X_test = lda.transform(X_test)

#classifier
from sklearn.linear_model import Lasso
regressor = Lasso(alpha=0.1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

accuracy = mse
precision = mae
recall = rmse
f1_score_value = r2

print("\nAccuracy : {:.4f}".format(accuracy))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score_value)



