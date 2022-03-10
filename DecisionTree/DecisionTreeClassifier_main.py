import numpy as np
import pandas as pd
import copy

class DecisionTreeClassifier: # essentially this class is the class of a Node. We will pass methods on this node to make it split and grow
  def __init__(self, X, Y, min_samples_split=None, # This will represent the root node 
               max_depth=None, depth=None, nodetype=None, loss_func='gini'):  # But same attributes will be re-used for child nodes
    
    # initialize left and right nodes as none, since the first instance will be the root node
    self.left = None
    self.right = None

    # initialize splitting features and values as None
    self.split_feature = None
    self.split_value = None

    # assigning arguments into instance attributes of the node (default max depth is 10, default min samples is 10)
    self.X = X
    self.Y = Y
    self.features = list(self.X.columns)
    self.md = max_depth if max_depth != None else 10
    self.mss = min_samples_split if min_samples_split != None else 10
    self.nodetype = nodetype if nodetype != None else 'root_node'

    # loss function as specified by hyperparameter
    if loss_func == 'gini':
      self.loss_func = 'gini'
    elif loss_func == 'entropy':
      self.loss_func = 'entropy'
    else:
      self.loss_func = None


    # we need this later to recursively add leaves to the tree and monitor its depth
    self.depth = depth if depth != None else 0 

    # count of observations falling under each unique label, and sort them in ascending
    self.counts = dict(self.Y.value_counts()) # e.g. {'versicolor': 747, 'virginica': 4785, 'dkfoasdkfo': 409}
    self.sorted_counts = list(sorted(self.counts, key = lambda x:x[1])) # sort by value count, not by label name

    # get loss score
    self.loss = self.gini_entropy() 

    # majority class of each node (also the predicted class for a node with no more children nodes)
    # Get the majority class if the node has at least 1 class, else return None
    self.y_pred = self.sorted_counts[0] if len(self.sorted_counts) > 0 else None 

    # number of obs in a node
    self.n = len(Y)

    # list of unique classes
    # self.classes = sorted(list(self.Y.unique())) # e.g. ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

  #function to get the gini/entropy of a node
  def gini_entropy(self): 
    
    class_counts = []
    for i in self.counts.items():
      class_counts.append((i[0], i[1])) # e.g. [('spam', 747), ('ham', 4825)]
    
    counts_list = [class_counts[i][1] for i in range(len(class_counts))] # e.g. [747, 4825]

    
    if self.loss_func == 'gini':
      return self.calculate_gini(counts_list) 
    elif self.loss_func == 'entropy':
      return self.calculate_entropy(counts_list)
    else:
      print("Error.")


  #function to get the entropy of a list of class counts
  @staticmethod
  def calculate_entropy(counts_list):

    n = sum(counts_list)

    if n == 0:
      return 0.0
    
    # probabilities of each class
    probs = []
    for i in counts_list:
      probs.append(i/n)

    # entropy formula
    entropy_list = []
    for i in probs:
      entropy_list.append(-i*np.log2(i+1e-9))

    entropy = sum(entropy_list)

    return entropy



  #function to get the gini of a list of class counts
  @staticmethod
  def calculate_gini(counts_list): 

    n = sum(counts_list)

    if n == 0:
        return 0.0

    # probabilities of each class
    probs = []
    for i in counts_list:
      probs.append(i/n)
    
    # gini formula 
    sum_probs = []
    for i in probs:
      sum_probs.append(i**2)

    sumprobs = sum(sum_probs)

    gini = 1 - sumprobs
    
    return gini # e.g. [0.45]

  # finds the best split given features and classes, using max gain of gini impurity algorithm
  def find_split(self):

    data = self.X.copy()
    data['classes'] = self.Y
    parent_loss = self.gini_entropy()
    gain = 0 # initialize gain (change in gini/entropy impurity) as zero

    # initialize best feature and best value as none
    split_feature = None
    split_value = None

    # iterate over each feature to find best feature for maximizing gini gain
    for i in self.features: # for each feature
        # drop NAs
        X_data = data.dropna()

        # sort by feature values in ascending order
        X_data = X_data.sort_values(i) 

        # get the mid (average) between 2 neighboring numerical values of the same feature
        x_mid = list(pd.Series(X_data[i].unique()).rolling(2).mean())[1:]
        # v = [1,1]
        # x_mid = (np.convolve(X_data[i].unique(), v, 'valid'))/2

        # iterate over each average of 2 neighboring values to find best split within this feature for maximizing gini/entropy gain
        for j in x_mid:

          left_counts = dict(X_data[X_data[i]<j]['classes'].value_counts()) # e.g. {class1 : 747, class2: 4825}
          right_counts = dict(X_data[X_data[i]>=j]['classes'].value_counts())

          left_list = [k[1] for k in left_counts.items()] # e.g. [747, 4825]
          right_list = [k[1] for k in right_counts.items()]

          if self.loss_func == 'gini':

            left_gini = self.calculate_gini(left_list)
            right_gini = self.calculate_gini(right_list)
            weighted_gini = (
              (left_gini*sum(left_list))/(sum(left_list)+sum(right_list))) + (
                  (right_gini*sum(right_list))/(sum(left_list)+sum(right_list))
                  ) # just the formula to get the weighted gini
            
            gini_gain = parent_loss - weighted_gini # getting the gini gain

            # Best split 
            if gini_gain > gain:
                split_feature = i
                split_value = j 
                gain = gini_gain
            else:
              pass
            
          elif self.loss_func == 'entropy':

            left_entropy = self.calculate_entropy(left_list)
            right_entropy = self.calculate_entropy(right_list)

            weighted_entropy = (
            (left_entropy*sum(left_list))/(sum(left_list)+sum(right_list))) + (
                    (right_entropy*sum(right_list))/(sum(left_list)+sum(right_list))
                    ) # just the formula to get the weighted entropy
          

            entropy_gain = parent_loss - weighted_entropy # getting the entropy gain

            # Best split 
            if entropy_gain > gain:
                split_feature = i
                split_value = j 
                gain = entropy_gain
            else:
              pass

    return (split_feature, split_value)

  def train(self):
    data = self.X.copy()
    data['classes'] = self.Y

    # based on hyperparameters, if max depth is not exceeded and if min samples are not activated, 
    # we split the tree further
    if (self.depth < self.md) and (self.n >= self.mss):

        # we override the initialized value of None for the root node with the new feature and value. This will be done recursively for child nodes
        split_feature, split_value = self.find_split()
        if split_feature != None: # basically we do not continue splitting the tree if the gini gain is 0 (since split_feature would be None)
            self.split_feature = split_feature
            self.split_value = split_value
            print('depth =',self.depth, self.nodetype, split_feature, split_value)

            # we assign the separated and split data into the left and right nodes along with other instance attributes, while updating the depth
            left_data = data[data[split_feature] <= split_value].copy()
            right_data = data[data[split_feature] > split_value].copy()

            left_node = DecisionTreeClassifier(
                X = left_data[self.features], # e.g. a dataframe with only the feature columns of the node without classes
                Y = pd.Series(left_data['classes'].values), # e.g. ['ham', 'spam', 'ham', 'spam', .....]
                min_samples_split = self.mss,
                max_depth = self.md,
                depth = self.depth + 1, # update depth
                nodetype='left_node')

            right_node = DecisionTreeClassifier(
                X = right_data[self.features], # e.g. a dataframe with only the feature columns of the node without classes
                Y = pd.Series(right_data['classes'].values), # e.g. ['ham', 'spam', 'ham', 'spam', .....]
                min_samples_split = self.mss,
                max_depth = self.md,
                depth = self.depth + 1, # update depth
                nodetype='right_node')
            
            # now we tell the base node that it has 2 child nodes, 1 left and 1 right, and apply the train function to the 2 of them
            self.left = left_node
            self.right = right_node
            self.left.train()
            self.right.train()

            # the train function will split the base node into left and right child nodes, and split the left and right child nodes into ...
            # ... more child left and right nodes, until either min_samples_split is met or max_depth is reached ...
            # ... then the algorithm stops. This way, the decision tree keeps growing by itself until a criterion is met


  # function to compile all predictions into a list and give that list as output
  def predict(self, X):

    preds = []

    for _, x in X.iterrows(): # for each row
        feature_vals = {}

        for i in self.features: # for each feature
            feature_vals.update({i: x[i]}) # e.g.  {'petal_length': 4.6, 'petal_width': 1.3, 'sepal_length': 6.6, 'sepal_width': 2.9}

        current = copy.deepcopy(self)

        stack = list()
        #stack.append(current)
        while (current != None) and (current.depth < current.md): # max depth
          #print(vars(current))
          
          split_feature = current.split_feature
          split_value = current.split_value
          #print(current.split_feature, current.split_value)
          if current.n < current.mss: # min samples split
              break

          # if less than split value goes to left node, else goes to right node
          #print(feature_vals.get(split_feature), split_value)
          if (split_value != None) and (feature_vals.get(split_feature) >= split_value):
              if current.right != None:
                  current = current.right
                  stack.append(current)
          elif (split_value != None):
              if current.left != None:
                  current = current.left
                  stack.append(current)
          
          if len(stack) > 0:
            current = stack.pop()
          else:
            break

          #while (current != None) and (len(stack) > 0):
            #current = stack.pop()


        preds.append(current.y_pred) # one prediction for one row

    return preds
