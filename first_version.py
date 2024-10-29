import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture
from joblib import dump,load
from xgboost import XGBClassifier

# 参数设定
num_iterations = 10
strategies = []
best_model = None
highest_auc_score = 0
data = pd.read_csv('train_gf_conbine.csv')
data.columns = data.columns.str.replace(r'[\[\]<]', '', regex=True)  # 移除不合法字符
data.columns = data.columns.astype(str)
data_splits = np.array_split(data, 10)
for i in range(0, num_iterations ):
    print(f"第{i+1}轮处理开始")
    data = data_splits[i]
    train_data, temp_data = train_test_split(data, test_size=0.4,random_state=i)
    test_data, validation_data = train_test_split(temp_data, test_size=0.5,random_state=i)


    # 2. 使用训练集训练
    X_train = train_data.drop(columns=['is_sa'])
    y_train = train_data['is_sa']
    # model = XGBClassifier(eval_metric='logloss')  # XGBoost模型
    # model.fit(X_train, y_train)
    # if(i == 0):
    #     dump(model, 'first.joblib')

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # 3. 使用测试集计算混淆矩阵以测试模型
    X_test = test_data.drop(columns=['is_sa'])
    y_test = test_data['is_sa']
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:\n{conf_matrix}")

    # 4. 使用高斯混合模型
    gmm = GaussianMixture(n_components=3, max_iter=200, tol=1e-3, init_params='random',random_state=i)
    gmm.fit(data)

    # 5. 将每个交易分配给最可能属于的高斯混合模型组件
    strategy_labels = gmm.predict(data)
    data_with_strategy = data.copy()
    data_with_strategy['strategy'] = strategy_labels
    strategies.append(strategy_labels)


    # 6. 选择最佳策略，定义为假阴性最高的策略
    # 计算每个策略的假阴性数量
    strategy_fns = {}  # 字典存储每个策略的假阴性数量

    for strategy in np.unique(strategy_labels):
        # 计算当前策略的预测结果
        current_strategy_data = data_with_strategy[data_with_strategy['strategy'] == strategy]
        current_y = current_strategy_data['is_sa']
        current_y_pred = model.predict(current_strategy_data.drop(columns=['is_sa', 'strategy']))

        # 计算假阴性（False Negatives）
        false_negatives_current = (current_y == 1) & (current_y_pred == 0)
        fn_count = false_negatives_current.sum()

        # 保存假阴性数量
        strategy_fns[strategy] = fn_count

    # 找到具有最高假阴性数量的策略
    best_strategy = max(strategy_fns, key=strategy_fns.get)
    print(f"最佳策略: {best_strategy}，假阴性数量: {strategy_fns[best_strategy]}")

    # 7. 在最佳策略的数据上使用SMOTE进行数据平衡
    best_strategy_data = data_with_strategy[data_with_strategy['strategy'] == best_strategy]
    X_best = best_strategy_data.drop(columns=['is_sa', 'strategy'])
    y_best = best_strategy_data['is_sa']

    # 检查正类样本数量
    if y_best.sum() < 2:  # 至少需要2个样本
        print("正类样本数量不足，无法应用SMOTE。")
    elif (i<9):
        smote = SMOTE(random_state=i, k_neighbors=3)
        X_smote, y_smote = smote.fit_resample(X_best, y_best)

        # 将合成的数据加入到下一轮的数据集中
        synthetic_data = pd.concat([pd.DataFrame(X_smote, columns=X_best.columns), pd.Series(y_smote, name='is_sa')],
                                   axis=1)
        data = pd.concat([data_splits[i+1], synthetic_data], ignore_index=True).drop_duplicates()

        # 9. 使用验证集来测试模型性能
    X_val = validation_data.drop(columns=['is_sa'])
    y_val = validation_data['is_sa']
    y_val_pred = model.predict(X_val)
    # 10. 计算ROC AUC分数并判断是否保存模型
    auc_score = roc_auc_score(y_val, y_val_pred)
    print(f"第{i + 1}轮的ROC AUC分数: {auc_score}\n")

    # 输出分类报告
    print(classification_report(y_val, y_val_pred))
    if (i > 0):
        first_model = load('first.joblib')
        first_y_val_pred = first_model.predict(X_val)
        auc_score = roc_auc_score(y_val, first_y_val_pred)
        print(f"静态第一次的ROC AUC分数: {auc_score}\n")
        print(classification_report(y_val, first_y_val_pred))

    # 更新最佳模型
    if auc_score > highest_auc_score:
        highest_auc_score = auc_score
        best_model = model

# 11. 保存最佳模型
dump(best_model, 'best_xgb_model.joblib')
print("数据处理和模型训练完成。最佳模型已保存。")
