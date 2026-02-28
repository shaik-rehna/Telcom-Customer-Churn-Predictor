import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=200),
                "XGBoost": XGBClassifier(eval_metric="logloss")
            }
            param = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300]
                }
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param
            )
            
            ## To get best model score from dict
            best_model_score = max(model_report.values())

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = trained_models[best_model_name]

            if best_model_score<0.7: # Because ROC-AUC baseline ~0.83
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            y_prob = best_model.predict_proba(X_test)[:,1]
            roc_auc = roc_auc_score(y_test, y_prob)
            return roc_auc

            
        except Exception as e:
            raise CustomException(e,sys)