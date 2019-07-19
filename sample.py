import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit

cv = 2

trainX= [[1], [2], [3], [4], [5]]

trainY = [1, 2, 3, 4, 5]

# these are the evaluation sets

testX = trainX 

testY = trainY

paramGrid = {"subsample" : [0.5, 0.8]}

fit_params={"early_stopping_rounds":42, 

            "eval_metric" : "mae", 

            "eval_set" : [[testX, testY]]}

model = xgb.XGBRegressor()

gridsearch = GridSearchCV(model, paramGrid, verbose=1 ,

         fit_params=fit_params,

         cv=TimeSeriesSplit(n_splits=cv).get_n_splits([trainX,trainY]))

gridsearch.fit(trainX,trainY)
