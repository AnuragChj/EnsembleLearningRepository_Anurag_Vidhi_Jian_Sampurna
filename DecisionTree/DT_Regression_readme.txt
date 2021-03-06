Decision Tree Regressor allows the user to create a decision tree and predict target variables.

The user starts by calling the function, inputting the features (dataset), the target variable, maximum depth of the tree and minimum sample split.
The function evaluates all possible values for different split points for best feature and value for the split.
The decision to choose the best split is based on the loss function : Mean Squared Error (MSE). The lowest possible MSE is selected and the corresponding split value is finalized
The data points lower than the splitting value go to the left node and the data points higher or equal to the splitting value go to the right node.
This process is repeated with the given subset on left and right nodes. The tree grows as the function is recursive.
At each split, the depth of the tree is constantly updated. Further splitting continues until maximum depth of the tree is reached or minimum sample split is not achieved.

The final decision tree when compared with Sklearn's implementation on the same dataset is very similar. 
