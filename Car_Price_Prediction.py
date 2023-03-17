import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle


df=pd.read_csv("E:/CarPricePredictionApp/car data.csv")
df.head()

df.shape

df.info()

df.columns


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()


final_dataset['Current_Year'] = 2023
final_dataset.head()


final_dataset['Years_Used'] = final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.head()


final_dataset.drop(['Year','Current_Year'],axis=1,inplace=True)
final_dataset.head()


print(final_dataset['Fuel_Type'].unique())
print(final_dataset['Seller_Type'].unique())
print(final_dataset['Transmission'].unique())
print(final_dataset['Owner'].unique())


final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()


final_dataset.corr()


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")



X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]



model = ExtraTreesRegressor()
model.fit(X,y)
model.feature_importances_


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train.shape , X_test.shape


rf = RandomForestRegressor()


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5,
                               verbose=2, random_state=0)


rf_random.fit(X_train,y_train)


rf_random.best_params_

predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


r2_score(y_test,predictions)


# open a file, where you want to store the data
file = open('car_Price_Prediction_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

