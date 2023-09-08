import random
from collections import defaultdict
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import csv
from sklearn import  ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



cakes = defaultdict(lambda: [])

def good_cake(baked,weight,temperature,speed,color):

    # rule 1
    if baked and weight and temperature<=200 and temperature>=180 and speed>=30 and speed<=50  :
        return True
    return False
    # rule 2
    # if color == 1 or color==2:
    #     if baked and weight and temperature<=200 and temperature>=180 and speed>=30 and speed<=50  :
    #         return True
    #     else:
    #         return False
    # elif color == 4:
    #     if baked and weight and temperature<=200 and temperature>=190 and speed>=35 and speed<=50  :
    #         return True
    #     else:
    #         return False
    #return False

    #rule 3
    # if baked and weight:
    #     if (temperature+speed)>=210 and (temperature+speed)<=250:
    #         return True
    #     elif temperature<=190 and temperature>=180 and speed>=35 and speed<=50:
    #         return True
    # return False




def generate_right_data():
    for i in range(500):
        temperature = random.randint(180, 200)
        speed = random.randint(30,55)
        baked = 1 #寫死
        weight = 1 #寫死
        size = random.randint(1,6)
        color = random.randint(1,3)
        flavor = random.randint(1,4)
        goodcake = good_cake(baked,weight,temperature,speed,color)
        # what we want the machine to learn
        cakes['baked'].append(baked)
        cakes['weight'].append(weight)
        cakes['speed'].append(speed)
        cakes['temperature'].append(temperature)

        # extra imformation that doesn't influence the result
        cakes['size'].append(size)
        cakes['color'].append(color)
        cakes['flavor'].append(flavor)
        # the result
        cakes['goodcake'].append(goodcake)

    data = pd.DataFrame.from_dict(cakes)
    return data

def generate_data():
    for i in range(500):
        temperature = random.randint(150, 230)
        speed = random.randint(30,55)
        baked = random.randint(0,1)
        weight = random.randint(0,1)
        size = random.randint(1,6)
        color = random.randint(1,3)
        flavor = random.randint(1,4)
        goodcake = good_cake(baked,weight,temperature,speed,color)

        # what we want the machine to learn
        cakes['baked'].append(baked)
        cakes['weight'].append(weight)
        cakes['speed'].append(speed)
        cakes['temperature'].append(temperature)

        # extra imformation that doesn't influence the result
        cakes['size'].append(size)
        cakes['color'].append(color)
        cakes['flavor'].append(flavor)
        # the result
        cakes['goodcake'].append(goodcake)

    data = pd.DataFrame.from_dict(cakes)
    return data


if __name__ == "__main__":
    data1 = generate_data()
    data2 = generate_right_data()
    data = pd.concat([data1,data2],axis=0)
    print('data',data)

    cake2 = defaultdict(lambda: [])
    cake2['baked'].append(1)
    cake2['weight'].append(1)
    cake2['speed'].append(35)
    cake2['temperature'].append(179.5)
    cake2['size'].append(3)
    cake2['color'].append(2)
    cake2['flavor'].append(2)
    cake2['goodcake'].append(False)
    df2 = pd.DataFrame.from_dict(cake2)

    data = pd.concat([data,df2],axis=0)
    print(data)
    
    y = data['goodcake']
    X = data.drop(['goodcake'], axis=1)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4)

    # # [1] 建立 random forest 模型
    # forest = ensemble.RandomForestClassifier(n_estimators = 100)
    # forest_fit = forest.fit(train_X, train_y)

    # # 預測
    # test_y_predicted = forest.predict(test_X)

    # df = pd.DataFrame(test_X)
    # df['Y'] = test_y
    # df['predict'] = test_y_predicted

    # # # 績效
    # accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    # print(accuracy)

    #[2] Decision tree training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    #testing and accuracy
    y_predict = model.predict(X_test)
    df = pd.DataFrame(X_test)
    df['Y'] = y_test
    df['predict'] = y_predict
    
    acc = accuracy_score(y_test, y_predict)
    print('accuracy',acc)
    
    #visualizing decision tree
    text_representation = tree.export_text(model)
    print(text_representation)

    # [3] KNN
    # knn = KNeighborsClassifier(n_neighbors=6)
    # knn.fit(train_X,train_y)
    # y_predict = knn.predict(test_X)
    # acc = accuracy_score(test_y, y_predict)
    # print('accuracy',acc)

    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4)
    # model = GaussianNB()
    # model.fit(train_X, train_y)
    # y_predict=model.predict(test_X)
    # acc = accuracy_score(test_y, y_predict)
    # print('accuracy',acc)

    #save the result
    df = pd.DataFrame(test_X)
    df['Y'] = test_y
    df['predict'] = y_predict
    #df.to_csv('goodcake_Bayes.csv')
    df.to_csv('goodcake_Decision_rule3.csv')
    #df.to_csv('goodcake_Random_rule3.csv')

