import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

data = pd.read_csv("/home/node01/text_match/tests/models_test/data/output_new_datasource_new.csv")

# 分析数据集信息
print("*" * 30 + " info " + "*" * 30)
print(data.info())
print(data.head())

y = data['is_or_not']
print("*" * 30 + " y " + "*" * 30)
print(y.head())

x = data[['score_sim_vec','score1','score2','score3','score4']]
print(x.info())
print(x.head())


x_dict_list = x.to_dict(orient='records')
print("*" * 30 + " train_dict " + "*" * 30)
print(pd.Series(x_dict_list[:5]))

dict_vec = DictVectorizer(sparse=False)
x = dict_vec.fit_transform(x_dict_list)
print("*" * 30 + " onehot编码 " + "*" * 30)
print(dict_vec.get_feature_names())
print(x[:5])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_test)

# 决策树分类器
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

print("*" * 30 + " 准确率 " + "*" * 30)
print(dec_tree.score(x_test, y_test))


# 随机森林分类器
# n_jobs: -1表示设置为核心数量
rf = RandomForestClassifier(n_jobs=-1)

# 网格搜索
# n_estimators: 决策树数目
# max_depth: 树最大深度
param = {
    "n_estimators": [120, 200, 300, 500, 800, 1200],
    "max_depth": [5, 8, 15, 25, 30]
}
# 2折交叉验证
search = GridSearchCV(rf, param_grid=param, cv=2)
print("*" * 30 + " 超参数网格搜索 " + "*" * 30)

start_time = time.time()
search.fit(x_train, y_train)



with open("data.pkl","wb") as f:
    pickle.dump(search,f)


print("耗时：{}".format(time.time() - start_time))
print("最优参数：{}".format(search.best_params_))

print("*" * 30 + " 准确率 " + "*" * 30)
print(search.score(x_test, y_test))

# result = search.predict([[0.97532535,1.0,0.513,1.0,0]])
result = search.predict([[0.9628693,0.4515,0.5165,0.49,0]])
print(result)
