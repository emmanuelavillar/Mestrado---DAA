# MULTICLASS
from sklearn.pipeline import Pipeline as PipelineIMB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as PipelineIMB

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from typing import Union, List, Tuple, Dict, Callable, Any

from sklearn.metrics import cohen_kappa_score, make_scorer


from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score, precision_score

import catboost
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import recall_score, f1_score, precision_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from typing import Union, Dict, List, Tuple, Any, Callable
from imblearn.pipeline import Pipeline as PipelineIMB
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


def roc_auc_multiclass_scorer(estimator, X, y):
        y_proba = estimator.predict_proba(X)
        return roc_auc_score(y, y_proba, multi_class='ovr', average='macro')

def cross_validation_models_set_class(
    abordagem: str, 
    preprocessor: Union[Callable, List[Tuple[str, Callable]]], 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> pd.DataFrame:
    
    classifiers: Dict[str, Any] = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000, multi_class='multinomial'),
        'K-Vizinhos Mais Próximos (KNN)': KNeighborsClassifier(),
        'Árvore de Decisão': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state, probability=True),
        'LightGBM': LGBMClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss'),
        'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0, auto_class_weights='Balanced'),
        'Balanced Random Forest': BalancedRandomForestClassifier(random_state=random_state),
        'EasyEnsemble': EasyEnsembleClassifier(random_state=random_state),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=random_state),
        'MLP (Neural Network)': MLPClassifier(random_state=random_state, early_stopping=True)
    }

    #cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=random_state)

    metrics = {
        'ROC-AUC':roc_auc_multiclass_scorer, #make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True),
        'Accuracy': make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score, average='macro'),
        'Recall': make_scorer(recall_score, average='macro'),
        'F1-Score': make_scorer(f1_score, average='macro'),
        'Kappa': make_scorer(cohen_kappa_score)
    }

    resultados = []

    for nome, classificador in classifiers.items():
        if abordagem == 'Oversampling':
            resampler = SMOTE(random_state=random_state)
        elif abordagem == 'Undersampling':
            resampler = TomekLinks()
        elif abordagem == 'Ambos':
            resampler = SMOTETomek(sampling_strategy='auto', smote=SMOTE(random_state=random_state), tomek=TomekLinks())
        else:
            resampler = None

        steps = [('Preprocessor', preprocessor)]
        if resampler:
            steps.append(('Resampler', resampler))
        steps.append(('Model', classificador))
        pipeline = PipelineIMB(steps)

        scores = []
        for metric_name, metric_func in metrics.items():
            score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=metric_func, n_jobs=-1).mean()
            scores.append(round(score, 3))

        resultados.append([nome] + scores)

    columns = ['Modelo'] + list(metrics.keys())
    df_resultados = pd.DataFrame(resultados, columns=columns)

    return df_resultados   
    classifiers: Dict[str, Any] = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'K-Vizinhos Mais Próximos (KNN)': KNeighborsClassifier(),
        'Árvore de Decisão': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state),
        'LightGBM': LGBMClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0, auto_class_weights='Balanced'),
        'Balanced Random Forest': BalancedRandomForestClassifier(random_state=random_state),
        'EasyEnsemble': EasyEnsembleClassifier(random_state=random_state),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=random_state),
        'MLP (Neural Network)': MLPClassifier(random_state=random_state, early_stopping=True)
    }

    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=random_state)
    #cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    kappa_scorer = make_scorer(cohen_kappa_score)
    
    resultados = []

    for nome, classificador in classifiers.items():
        if abordagem == 'Oversampling':
            resampler = SMOTE()
        elif abordagem == 'Undersampling':
            resampler = TomekLinks()
        elif abordagem == 'Ambos':
            resampler = SMOTETomek(sampling_strategy='auto', smote=SMOTE(), tomek=TomekLinks())
        else:
            resampler = None

        steps = [('Preprocessor', preprocessor)]
        if resampler:
            steps.append(('Resampler', resampler))
        steps.append(('Model', classificador))
        pipeline = PipelineIMB(steps)

        metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', kappa_scorer]
        scores = [cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1).mean() 
                  for metric in metrics]

        resultados.append([nome] + [round(score, 3) for score in scores])

    columns = ['Modelo', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']
    df_resultados = pd.DataFrame(resultados, columns=columns)

    return df_resultados


def hyperparameter_optimization(
    resampler: Union[None, Any],
    preprocessor: Union[Callable, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modelos: Dict[str, tuple]
) -> pd.DataFrame:

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = []

    scoring = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro')
    }

    for model_name, (model, params) in modelos.items():
        steps = [('Preprocessor', preprocessor)]
        if resampler:
            steps.append(('Resampler', resampler))
        steps.append(('model', model))
        pipe = PipelineIMB(steps)

        grid_search = GridSearchCV(
            pipe,
            param_grid=params,
            cv=tscv,
            scoring=scoring,
            refit='f1_macro',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        scores = {
            'model': model_name,
            'F1_macro': grid_search.cv_results_['mean_test_f1_macro'][grid_search.best_index_],
            'Recall_macro': grid_search.cv_results_['mean_test_recall_macro'][grid_search.best_index_],
            'Precision_macro': grid_search.cv_results_['mean_test_precision_macro'][grid_search.best_index_],
            'Best_params': grid_search.best_params_
        }

        best_score.append(scores)

    return pd.DataFrame(best_score)



def curva_roc(X_test: pd.DataFrame, y_test: pd.Series, pipe_over):
    y_proba = pipe_over.predict_proba(X_test)
    classes = pipe_over.classes_
    
    y_test_bin = label_binarize(y_test, classes=classes)
    
    plt.figure(figsize=(6, 4))
    
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f'Classe {cls} (AUC={auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def eq_reg_log(pipe, X_train):
    model = pipe.named_steps['model']
    preprocessor = pipe.named_steps['Preprocessor']

    X_transformed = preprocessor.transform(X_train)

    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = X_train.columns

    if model.coef_.shape[0] == 1:
        # Binário
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        equation = f"log(P(y=1)/(1-P(y=1))) = {intercept:.3f} "
        for i, coef in enumerate(coefficients):
            equation += f"+ ({coef:.3f} * {feature_names[i]}) "
        print("Equação da Regressão Logística (binária):")
        print(equation)
    else:
        # Multiclasse
        for class_idx, class_coef in enumerate(model.coef_):
            intercept = model.intercept_[class_idx]
            equation = f"log(P(y={class_idx}) / P(y=ref)) = {intercept:.3f} "
            for i, coef in enumerate(class_coef):
                equation += f"+ ({coef:.3f} * {feature_names[i]}) "
            print(f"Equação da Regressão Logística (classe {class_idx}):")
            print(equation)
            print()













