DEBUG = True
HFO_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
type_names_to_run = ['RonO', 'RonS', 'Fast RonO', 'Fast RonS']

from models import naive_bayes, random_forest, svm_m, balanced_random_forest, xgboost
models_to_run = ['Naive Bayes', 'Random Forest', 'SVM', 'Balanced RF', 'XGBoost']
models_to_run_obj = [naive_bayes, random_forest, svm_m, balanced_random_forest, xgboost]



