<h2> Final Project: Yandex Real Estate Price Prediction
 </h2>

![alt text](https://github.com/abuser11/e2e_project_2024/blob/master/images/spb_resal_estate.jpg)

**Dataset Description:**

The dataset contains various columns of information related to real estate properties:

1. **Last Price:** The most recent price at which the property was listed or sold.
2. **Floor:** The floor number on which the property is located.
3. **Rooms:** The number of rooms available in the property.
4. **Area:** The total area or size of the property.
5. **Kitchen Area:** The area specifically designated for the kitchen within the property.
6. **Living Area:** The area intended for living spaces such as the living room, bedrooms, etc.
7. **Agent Fee:** The fee charged by the agent or agency involved in the property transaction.
8. **Renovation:** Indicates whether the property has undergone any recent renovations or improvements.
9. **Building ID:** An identifier associated with the specific building or property.

***
**Data used for the project:**

* [St. Petersb and Leningrad Oblast data (2016-2018)](https://github.com/abuser11/e2e_project_2024/blob/master/data/spb.real.estate.archive.sample5000.tsv)

* [Cleaned Data](https://github.com/abuser11/e2e_project_2024/blob/master/data/cleaned_dataset.csv)
***

<h2> Work description </h2>

**[lab1_EDA_and_Visualization.ipynb](https://github.com/abuser11/e2e_project_2024/blob/master/notebooks/lab1_1_EDA_and_Visualzation.ipynb) contains the preprocesing, data cleaning from outliers and EDA of the real estate dataset. Besides, it answers following questions:**

1. Calculate median and mean prices for apartments for rent after cleaning the data in St.Petersburg without Leningrad Oblast. Which of the statistics changed more and why?
2. Calculate median and mean prices for apartments for sell before cleaning the data.
3. Find houses with the most cheapest and most expensive price per sq m in St. Petersburg without Leningrad Oblast after cleaning outliers.
4. Find the most expensive and the most cheapest apartment in St. Petersburg after cleaning outliers.
5. Calculate how many years does it take to cover all money spent on buying apartment by renting it. Find houses in which it's the most effective to invest in buying if you plan to rent and houses in which it will be the most ineffective.

**Examples of Visualization (clustering):**

![alt text](https://github.com/abuser11/e2e_project_2024/blob/master/images/plot1.png)
![alt text](https://github.com/abuser11/e2e_project_2024/blob/master/images/plot2.png)

After preprocessing of the data we created the model itself, as a metric of the algorithm performance MSE was chosen. The best scaling techincue and ML algorithm are **Standard Scaling** and __DecisionTreeRegressor__ 

```
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.fit_transform(X_valid)
y_train = sc_y.fit_transform(y_train)
y_valid = sc_y.fit_transform(y_valid)

dt = DecisionTreeRegressor(max_depth=40, min_samples_leaf=15, max_features=4, min_samples_split=2, splitter = 'best')
```	

**To find optimal hyperparametrs for the model RandomizedSearchCV approach was used:**

```
param_dist = {
    'max_depth': [10, 15, 20, 25, 30, 40, 50, 60, 70],
    'min_samples_leaf': [5, 10, 15, 20, 25, 30, 35, 40],
    'max_features': [2, 3, 4],
    'min_samples_split': [2, 3, 4, 5, 6, 7],
    'splitter': ['best', 'random'],
    'criterion': ['mse', 'mae'],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
}


dt = DecisionTreeRegressor()
random_search = RandomizedSearchCV(dt, param_dist, n_iter=50, scoring='neg_mean_squared_error', cv=5)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

predictions = best_model.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)
mse = mean_squared_error(y_valid, predictions)
rmse = np.sqrt(mse)

print('Best parameters:', random_search.best_params_)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
```	

**Best parameters**: 
- 'splitter': 'best', 
- 'random_state': 7, 
- 'min_weight_fraction_leaf': 0.0, 
- 'min_samples_split': 2, 
- 'min_samples_leaf': 15, 
- 'max_features': 4,
- 'max_depth': 40,
- 'criterion': 'mse'
---
And the results for the model:
- **MAE**: 0.35
- **MSE**: 0.38
- **RMSE**: 0.62
---

**Model prediction on the data:**

![alt text](https://github.com/abuser11/e2e_project_2024/blob/master/images/model.png)
![alt text](https://github.com/abuser11/e2e_project_2024/blob/master/images/model2.png)


<h2> Additional information </h2>

**During this project I learn how to deal with Docker containers and remote machine. Besides, I improved my skills with github and Flask**

You may connect to may virtual machine using following username and IP:

```	
ssh osanov@51.250.99.187
```	

The Dockerfile content/algorythm of actions:

```	
FROM ubuntu:22.04
MAINTAINER Artem Osanov
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
```	
**Information about remote machine:**
```	
app.run(debug = True, port = 5444, host = '0.0.0.0')
```	
In order to run virtual machine port we should use this code after:
```	
sudo apt install ufw
sudo ufw allow 5444 
```	

Then build and run the docker container

```	
#My image:
#osanov/gsom_e2e_2024:v.0.1

#Basic algorithm:
docker build -t <your login>/<directory name>:<version> .     
docker run --network host -it <your login>/<directory name>:<version> /bin/bash
docker run --network host -d <your login>/<directory name>:<version>   
docker ps 
docker stop <container name> 
```	
