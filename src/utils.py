import os
import sys

import numpy as np 
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            test_model_score = roc_auc_score(y_test, y_test_prob)

            report[model_name] = test_model_score
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)