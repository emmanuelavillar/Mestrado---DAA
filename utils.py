import shap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def shap_(preprocessor, X, X_test, pipe_over): 
    shap.initjs() 

    feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]

    X_shap = preprocessor.transform(X)
    test_shap = preprocessor.transform(X_test)
    explainer =  shap.LinearExplainer(pipe_over.named_steps['model'], X_shap)
    shap_values = explainer.shap_values(test_shap)

    shap.summary_plot(shap_values, test_shap, feature_names = feature_names)

def shap_tree(preprocessor, X, X_test, pipe_over): 
    shap.initjs() 

    feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]

    X_shap = preprocessor.transform(X)
    test_shap = preprocessor.transform(X_test)
    explainer =  shap.TreeExplainer(pipe_over.named_steps['model'], X_shap)
    shap_values = explainer.shap_values(test_shap)

    shap.summary_plot(shap_values, test_shap, feature_names = feature_names)


def feature_importance(pipe):
    modelo = pipe.named_steps['model']
    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    importancia = pd.DataFrame({
        'Feature': feature_names,
        'Importance': modelo.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importancia.head(20))
    plt.title("Importância das Features")
    plt.show()

