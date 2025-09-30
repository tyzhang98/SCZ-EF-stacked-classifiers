import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from optuna.integration import SkoptSampler
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
import pickle
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import StandardScaler

# 读取这些变量

with open('./data/variables-huafen.pkl', 'rb') as file:
    X_resampled, y_resampled = pickle.load(file)
exclusive = ['PANSS_Negative1', 'PANSS_Positive1', 'PANSS_Affective1', 'PANSS_Cognitive1', 'PANSS_Negative2', 'PANSS_Positive2', 'PANSS_Affective2', 'PANSS_Cognitive2', 'PANSS_Negative3', 'PANSS_Positive3', 'PANSS_Affective3', 'PANSS_Cognitive3']
X , y = X_resampled[list(set(X_resampled.columns.tolist())-set(exclusive))],y_resampled.to_numpy()

# 步骤3: 定义目标函数进行超参数优化
def optuna_opti(trial, model_type, x, y):
    # 在函数内部使用Loguru的logger记录Optuna日志
    # logger.info(f"Optimizing model {model_type}...")
    if model_type == 'SVM':
        params = {'random_state': 0, 'probability': True}
        params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        params['C'] = trial.suggest_float('C', 0.001, 0.01) #跟分类2和分类3有区别
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])
        clf = SVC(**params)
    elif model_type == 'AdaBoost':
        params = {'random_state': 0}
        params['n_estimators'] = trial.suggest_int('n_estimators', 1, 100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 0.1)
        clf = AdaBoostClassifier(**params)
    elif model_type == 'RandomForest':
        params = {'random_state': 0, 'n_jobs': -1}
        params['n_estimators'] = trial.suggest_int('n_estimators', 10, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 1, 40)
        clf = RandomForestClassifier(**params)
    elif model_type == 'Stacking':
        params = {'random_state': 0, 'probability': True}
        params['C'] = trial.suggest_float('C', 0.001, 10000)
        clf_svm = SVC(**params)
        params = {'random_state': 0}
        params['n_estimators'] = trial.suggest_int('n_estimators1', 1, 100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 1)
        clf_ada = AdaBoostClassifier(**params)
        params = {'random_state': 0}
        params['n_estimators'] = trial.suggest_int('n_estimators2', 1, 100)
        params['max_depth'] = trial.suggest_int('max_depth', 1, 100)
        clf_rf = RandomForestClassifier(**params)
        estimators = [
            ('RF', clf_rf),
            ('SVM', clf_svm),
            ('Ada', clf_ada),
        ]
        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(random_state=1), n_jobs=-1, cv=3
        )
    
    clf.fit(x, y)
    train_acc = balanced_accuracy_score(y, clf.predict(x))
    return train_acc

# 定义模型列表
models = {
    "SVM": SVC,
    "AdaBoost": AdaBoostClassifier,
    "RandomForest": RandomForestClassifier,
    "Stacking": StackingClassifier,
}

def get_best_stacking(best_params):
    params = {'random_state': 0, 'probability': True, 'C': best_params['C']}
    clf_svm = SVC(**params)
    params = {'random_state': 0, 'n_estimators' : best_params['n_estimators1'], 'learning_rate' : best_params['learning_rate']}
    clf_ada = AdaBoostClassifier(**params)
    params = {'random_state': 0, 'n_estimators' : best_params['n_estimators2'], 'max_depth' : best_params['max_depth'],}
    clf_rf = RandomForestClassifier(**params)
    estimators = [
        ('RF', clf_rf),
        ('SVM', clf_svm),
        ('Ada', clf_ada),
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(random_state=1), n_jobs=-1, cv=3
    )
    return clf

epoch = 100

inner_metrics_list = []  # 存储内循环性能指标的列表
out_metrics_list = []   # 存储外循环性能指标的列表
hold_metrics_list = []  # 存储holdout性能指标的列表
hold_params_list = []  # 存储holdout性能指标的列表

# 设置种子
np.random.seed(42)

# 生成100个随机数
random_numbers = np.random.rand(100)
random_numbers = [int(x * 1000) for x in random_numbers]

global_best_item = {}
global_best_item_list = []
for model_name, model_type in models.items():
    # 步骤4: 执行重复嵌套交叉验证并进行超参数优化
    best_hyperparameters = []
    global_best_acc = 0
    for _ in tqdm(range(epoch)):
        best_auc = 0
        best_params = {}
        best_params_list = []
        scaler = StandardScaler()
        # 步骤1: 分层随机分割数据集
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_numbers[_])
        train_indices = X_train.index
        test_indices = X_holdout.index
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
        X_train.index = train_indices
        X_holdout = pd.DataFrame(scaler.transform(X_holdout), columns = X_train.columns)
        X_holdout.index = test_indices
        # # 获取划分后的索引
        joblib.dump(scaler,f'./StandarScaler/{model_name}{_}.pkl')
        with pd.ExcelWriter(f'./split_csv/{model_name}_indices{_}.xlsx') as writer:
            temp1 = deepcopy(X_train)
            temp1['Y'] = y_train
            temp2 = deepcopy(X_holdout)
            temp2['Y'] = y_holdout
            X_resampled[exclusive].iloc[train_indices].to_excel(writer, sheet_name='Train Data')
            X_resampled[exclusive].iloc[test_indices].to_excel(writer, sheet_name='Test Data')
            temp1.to_excel(writer, sheet_name='Hold Train')
            temp2.to_excel(writer, sheet_name='Hold Test')

        X_train = X_train.to_numpy()
        X_holdout = X_holdout.to_numpy()
        
        # 步骤2: 定义内部和外部交叉验证
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # 外部 5 划分
        for outer_train_idx, outer_test_idx in outer_cv.split(X_train, y_train):
            X_train_outer, X_test_outer = X_train[outer_train_idx], X_train[outer_test_idx]
            y_train_outer, y_test_outer = y_train[outer_train_idx], y_train[outer_test_idx]

            # 内部 3 划分
            for inner_train_idx, inner_test_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner, X_test_inner = X_train[inner_train_idx], X_train[inner_test_idx]
                y_train_inner, y_test_inner = y_train[inner_train_idx], y_train[inner_test_idx] 

                algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'n_initial_points': 10, 'acq_func': 'EI'})
                study = optuna.create_study(sampler=algo, direction="maximize")
                study.optimize(lambda trial: optuna_opti(trial, model_name, X_train_inner, y_train_inner), n_trials=100, show_progress_bar=True)
                best_params_list.append(study.best_trial)
                best_trial = study.best_trial

                if best_trial.value > best_auc:
                    best_auc = best_trial.value
                    best_params = best_trial.params

                # 计算内循环性能指标
                if model_name == "SVM":
                    best_params['probability'] = True
                if model_name=='Stacking':
                    clf = get_best_stacking(best_params)
                else:
                    clf = model_type(**best_params)
                clf.fit(X_train_inner, y_train_inner)
                y_pred_prob = clf.predict_proba(X_test_inner)[:, 1]
                inner_metrics_list.append(np.array([y_test_inner, y_pred_prob]))

            # 计算外循环性能指标
            if model_name == "SVM":
                best_params['probability'] = True
            if model_name=='Stacking':
                clf = get_best_stacking(best_params)
            else:
                clf = model_type(**best_params)
            clf.fit(X_train_outer, y_train_outer)
            y_pred_prob = clf.predict_proba(X_test_outer)[:, 1]
            out_metrics_list.append(np.array([y_test_outer, y_pred_prob]))

        # 计算holdout性能指标
        if model_name == "SVM":
            best_params['probability'] = True
        if model_name=='Stacking':
            clf = get_best_stacking(best_params)
        else:
            clf = model_type(**best_params)
        clf.fit(X_train, y_train)

        # Save the model
        best_model_item = {'clf': clf}
        with open(f'./model_history_holdout/{model_name}_{_}.pkl', 'wb') as f:
            pickle.dump(best_model_item, f)

        y_pred_prob = clf.predict_proba(X_holdout)[:, 1]
        hold_metrics_list.append(np.array([y_holdout, y_pred_prob]))
        hold_params_list.append(best_params)

        temp_acc = balanced_accuracy_score(y_holdout, (y_pred_prob > 0.5).astype(int))

        if temp_acc > global_best_acc:
            global_best_item["params"] = best_params
            global_best_item["clf"] = clf
            global_best_item["eval"] = hold_metrics_list[-1]
            global_best_item["index"] = _
            global_best_acc = temp_acc


        # 将优化结果转化为DataFrame
        df = study.trials_dataframe()
        # 将DataFrame保存到文件中
        df.to_csv(f'./optuna/optuna_{model_name}_{_}.csv', index=False, encoding = 'utf_8_sig')
    global_best_item_list.append(global_best_acc)

# 将列表保存到文件
with open('metrics/inner_metrics_list.pkl', 'wb') as f:
    pickle.dump(inner_metrics_list, f)
# 将列表保存到文件
with open('metrics/out_metrics_list.pkl', 'wb') as f:
    pickle.dump(out_metrics_list, f)
# 将列表保存到文件
with open('metrics/hold_metrics_list.pkl', 'wb') as f:
    pickle.dump(hold_metrics_list, f)
# 将列表保存到文件
with open('metrics/best_params_list.pkl', 'wb') as f:
    pickle.dump(best_params_list, f)
# 将列表保存到文件
with open('metrics/hold_params_list.pkl', 'wb') as f:
    pickle.dump(hold_params_list, f)
# 将列表保存到文件
with open('metrics/best_model_item.pkl', 'wb') as f:
    pickle.dump(global_best_item, f)
with open('metrics/best_model_item_list.pkl', 'wb') as f:
    pickle.dump(global_best_item_list, f)


    print("模型训练全部完成！")