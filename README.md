# CART decision tree and random forest
Practice of adaboost decision stump and bagging.

## How to run

```
python RidgeRegression.py DataPath lambda
```
```
python RidgeRegression+Bagging.py DataPath Lambda Iteration
```
```
python AdaboostStump.py TrainData TestData Iteration
```

## Description

### RidgeRegression.py

Implemented ridge regression on my own, which is linear regression regularized by L2 regularizer. Also imported sklearn's ridge regression, and the result is slightly different, maybe due to that sklearn's ridge regression has x_0 built in already.

### RidgeRegression+Bagging.py

Run bagging on top of ridge regression.

### AdaboostStump.py

Run adaboost(Adaptive Boosting) with decision stump, which is an aggregation model that calculate the voting weight on the fly! Every iteration, adaboost will reweight the data, using bagging, according to if it was classified correctly last iteration, incorrectly classified data will be given bigger weight, and correctly classified data will be given smaller weight. Then perform the base algorithm( in our case, decision stump), and then calculate this new model's voting weight with it's error. Continue this process until the desired iteration is achieved. Adaboost is guarenteed to reach E_in 0 in O(log N) iterations, however, since adaboost is an aggregation model, it doesn't get overfitting too much.

## Built With

* Python 3.6.0 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)