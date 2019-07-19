import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# GBDT(Gradient Boosting Decision Tree)
import xgboost as xgb # 2016年登場？
from xgboost import XGBClassifier
#import lightgbm as lgb # 2017年登場？
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier # 2018年登場？

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

def model_rf(X_train, X_test, y_train, y_test):
    # GSで検証するパラメータ
    param_grid = {
        'n_estimators': [10, 11, 12, 13],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'random_state': [0]
    }

    # GS内でのearlystoppingを可能とする？
    fit_params = {
        #'early_stopping_rounds': 100,
        #'eval_set': [[X_test, y_test]],
    }
    
    # モデルの生成
    model = RandomForestClassifier()
    gscv = GridSearchCV(
        model, # 使用するモデル
        param_grid=param_grid, # GridSearchで検証したいパラメータ
        #fit_params=fit_params, # GridSearchのパラメータ
        cv=3, # 交差検証分割数（標準は3）
        n_jobs=-1, # 使用するCPUコア数（-1なら全コア使用）
        verbose=0, # 進捗の出力（0なら非表示、1なら表示、2なら全表示）
        return_train_score=False, # 学習データに対する評価を行うかどうか
    )

    # 学習
    #model.fit(X_train, y_train)
    gscv.fit(X_train, y_train)#, early_stopping_rounds=1)

    # グリッドサーチの検証
    result = pd.DataFrame(gscv.cv_results_)
    result.sort_values(by='rank_test_score', inplace=True)
    #print(result[['rank_test_score', 'params', 'mean_test_score']])
    result.to_csv('./result_rf.csv')

    return gscv.best_estimator_

def model_xgb(X_train, X_test, y_train, y_test):
    params = {'subsample': [0.5, 0.8]}
    fit_params = {
        'early_stopping_rounds': 42,
        'eval_metric': 'mae',
        'eval_set': [[X_test, y_test]]}
    model = xgb.XGBRegressor()
    gscv = GridSearchCV(model, params, verbose=1, fit_params=fit_params, cv=3)
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_

def model_xgb_sklearn(X_train, X_test, y_train, y_test):
    param_grid = {
        #'eta': [0.2, 0.3, 0.4], # 学習の重み
        #'gamma': [0, 1, 2],
        'max_depth': [5, 6, 7], # 木の深さ
        'min_child_weight': [1, 2, 3], # 増加することで過学習を減らす？
        #'max_delta_step': [0, 1, 2],
        #'subsample': [0.8, 0.9, 1], # 使用するオブジェクトの割合
        #'colsample_bytree': [0.8, 0.9, 1],
        #'colsample_bylevel': [0.8, 0.9, 1],
    }

    fit_params = {
        'early_stopping_rounds': 100,
        'eval_set': [[X_test, y_test]],
    }

    # モデルの生成
    model = XGBClassifier(verbose=0)
    gscv = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        verbose=0,
        return_train_score=False
    )

    # 学習
    gscv.fit(X_train, y_train)

    # グリッドサーチの検証
    result_df = pd.DataFrame(gscv.cv_results_)
    result_df.sort_values(by='rank_test_score', inplace=True)
    #print(result_df[['rank_test_score', 'params', 'mean_test_score']])
    result_df.to_csv('./result_xgb.csv')

    return gscv.best_estimator_

def model_lgb(X_train, X_test, y_train, y_test):
    # LightGBMが扱えるようにデータを加工
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test)

    # パラメータ
    params = {
        'tast': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'}
    }

    # モデルを学習
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=dtest,
        early_stopping_rounds=10
    )

    return model

def model_lgb_sklearn(X_train, X_test, y_train, y_test):
    param_grid = {
            'max_depth': [-1],
    }

    fit_params = {
        'early_stopping_rounds': 100,
        'eval_set': [[X_test, y_test]]
    }
    
    model = LGBMClassifier(verbose=0)
    gscv = GridSearchCV(
        model,
        param_grid=param_grid,
        fit_params=fit_params,
        cv=3,
        verbose=0,
        return_train_score=False
    )
    gscv.fit(X_train, y_train)

    # グリッドサーチの検証
    result_df = pd.DataFrame(gscv.cv_results_)
    result_df.sort_values(by='rank_test_score', inplace=True)
    result_df.to_csv('./result_lgb.csv')

    return gscv.best_estimator_

def model_cb(X_train, X_test, y_train, y_test):
    # CatBoostが扱えるようにデータを加工
    dtrain = cb.DMatrix(X_train, y_train)
    dtest = cb.DMatrix(X_test, y_test)

    return model

def model_cb_sklearn(X_train, X_test, y_train, y_test):
    param_grid = {
        'depth': [4, 5, 6, 7, 8, 9, 10]    
    }

    fit_params = {
        'early_stopping_rounds': 100,
    }
    
    model = CatBoostClassifier(iterations=100)
    gscv = GridSearchCV(
        model,
        param_grid=param_grid,
        fit_params=fit_params,
        cv=3,
        verbose=0,
    )
    gscv.fit(X_train, y_train)

    result_df = pd.DataFrame(gscv.cv_results_)
    result_df.sort_values(by='rank_test_score', inplace=True)
    result_df.to_csv('./result_cb.csv')

    return gscv.best_estimator_

def model_dnn(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(activation="relu"))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    param_grid = {
        'activation': ['relu', 'sigmoid'],
        'optimizer': ['adam', 'adagrad'],
        'out_dim': [100, 200],
        'nb_epoch': [10, 25],
        'batch_size': [5, 10]
    }
    
    model = KerasClassifier(build_fn=model, verbose=0)
    gs = GridSearchCV(estimator=model, param_grid=param_grid)
    gs.fit(X_train, y_train)


if __name__ == '__main__':
    # データの用意
    df = datasets.load_breast_cancer()

    # データの確認
    #print(type(df))

    # データの分割
    X, y = df.data, df.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
        stratify=y
    )

    # データの確認（numpy配列になっている）
    #print(type(X_train))
    #print(type(X_test))
    #print(type(y_train))
    #print(type(y_test))

    models = {
        'RandomForest': model_rf(X_train, X_test, y_train, y_test),
        'XGBoost': model_xgb_sklearn(X_train, X_test, y_train, y_test),
        'LightGBM': model_lgb_sklearn(X_train, X_test, y_train, y_test),
        'CatBoost': model_cb_sklearn(X_train, X_test, y_train, y_test),
        #'DNN': model_dnn(X_train, X_test, y_train, y_test)
    }

    # 予測と検証
    print()
    for model, name in zip(models.values(), models.keys()):
        # 検証用データが各クラスに分類される確率を計算
        pred = model.predict(X_test)

        # 閾値0.5で0, 1に丸める
        y_pred = np.where(pred > 0.5, 1, 0)

        # 精度（Accuracy）を検証
        acc = accuracy_score(y_test, y_pred)

        print(name, '\t正解率: ', acc)
    print()
