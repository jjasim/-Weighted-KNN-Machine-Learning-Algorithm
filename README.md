# KNN Machine Learning Algorithm
My implementation of both weighted and unweighted KNN algorithms

## Unweighted KNN
This is a generic implementation of a brute-force KNN algorithm. It uses euclidean distances to determine the k-closest neighbours as shown below:

<img src="https://github.com/jjasim/Weighted-KNN-Machine-Learning-Algorithm/blob/main/images/unweighted.png" width="380" height="100">

## Weighted KNN
For the weighted KNN, the weights are on the features of a particular dataset. 
So one can bias the effect of some features over others.

This does make sense in some cases. For example, if you were to classify between cats and dogs, and you had the variables "has_fur" and "can_bark", you probably might want to assign a greater weight to the latter variable. Afterall, both dogs and cats have fur, but only dogs can bark.

To implement this weighted effect, I implemented a weighted euclidean distance to gauge the weighted distance between neighbours as shown below:

<img src="https://github.com/jjasim/Weighted-KNN-Machine-Learning-Algorithm/blob/main/images/weighted.png" width="350" height="100">

## Possible improvements
Try to add gradient descent to figure out the optimal weights that minimises errors. 
