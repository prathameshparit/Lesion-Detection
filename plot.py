# sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, auc, roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm, model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, \
    gaussian_process

# load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from math import sqrt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from pandas.core.indexing import convert_missing_indexer
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




def comparitive_plot(confidence_SVM,
                     confidence_KNN,
                     confidence_DT,
                     confidence_ANN,
                     confidence_CNN,
                     confidence_Hybrid):
    model_columns = []
    model_compare = pd.DataFrame(columns=model_columns)
    model_Name = [
        'SVM',
        'KNN',
        'DT',
        'ANN',
        'CNN',
        'Hybrid'
    ]
    accs = [
        confidence_SVM,
        confidence_KNN,
        confidence_DT,
        confidence_ANN,
        confidence_CNN,
        confidence_Hybrid
    ]
    count = 0
    row_index = 0
    for i in model_Name:
        model_name = model_Name[count]
        model_compare.loc[row_index, 'Name'] = model_name
        model_compare.loc[row_index, 'Accuracies'] = accs[count]

        count += 1
        row_index += 1

    model_compare.sort_values(by=['Accuracies'], ascending=False, inplace=True)
    print(model_compare)

    model_compare.to_csv('compare.csv', encoding='utf-8-sig')


    plt.subplots(figsize=(15, 6))
    sns.barplot(x="Name", y="Accuracies", data=model_compare, palette='hot', edgecolor=sns.color_palette('dark', 7))
    plt.xticks(rotation=90)
    plt.title('Accuracies Comparison')
    plt.savefig('static/assets/img/pred.jpg')

    return model_compare

# confidence_SVM = 0.99
# confidence_KNN = 0.99
# confidence_DT = 0.99
# confidence_ANN = 0.67
# confidence_CNN = 0.67
# confidence_Hybrid = 0.67
#
# comparitive_plot(confidence_SVM,
#                  confidence_KNN,
#                  confidence_DT,
#                  confidence_ANN,
#                  confidence_CNN,
#                  confidence_Hybrid)