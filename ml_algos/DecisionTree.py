import numpy as np
from collections import Counter
from typing import Set, Dict, Tuple

class TreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, threshold=None) -> None:
        
        # Feature index for train data
        self.feature = feature
        
        # The node value
        self.value = value

        # Left Node
        self.left = left
        # Right Node
        self.right = right
        
        # Threshold to decide if we go for left or right path
        self.threshold = threshold
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None) -> None:
        self.min_samples_split = min_samples_split
    
        self.max_tree_depth = max_depth 
        self.n_features = n_features
        self.root=None

    
    def fit(self, X, y):
        # set n_features using X features len if it is not defined.

        if not hasattr(X, 'shape') or len(X.shape) != 2:
            raise Exception("X object should contain a valid shape")
        

        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        self.root = self._expand_tree(X, y)

    def _expand_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y)) # len of unique labels

        # check stops conditions
        # Stop if we reach depth level or we reach a single labes or the min n_samples 
        if depth >= self.max_tree_depth or n_labels ==1 or n_samples< self.min_samples_split:
            
            label = self._most_common_label(y)

            return TreeNode(value=label)

        # Just get indexes in random order
        feature_indexes =np.random.choice(n_features, n_features, replace=False)

        # Find the best split
        best_feature_index, best_threshold = self._best_split(X, y, feature_indexes)

        # Expand the left and right sub_tree
        left_indexes, right_indexes = self._split(X[:, best_feature_index], best_threshold)

        left = self._expand_tree(X[left_indexes, :], y[left_indexes], depth=depth+1)
        right = self._expand_tree(X[right_indexes, :], y[right_indexes], depth=depth+1)
        
        return TreeNode(feature=best_feature_index, left=left, right=right, threshold=best_threshold)



    def _best_split(self, X: np.ndarray, y: np.array, feature_indexs: Set[str]) -> Tuple[str, float]:
        """
        For all columns in X:
            for all unique values (threshold) in columns:
                find the overall best information gain and threshold and return it

        Returns
            Tuple(split_index, threshold): Feature index of the columns with best IG, and the threshold
        """

        best_gain = -1
        split_index, split_threshold = None, None

        for col_index in feature_indexs:
            # Get unique values  in X column as threshold:
            X_column = X[:, col_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold=threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = col_index
                    split_threshold = threshold
            
        return split_index, split_threshold


    def _information_gain(self, y, X_column, threshold):
        """
        Calculate Information Gain:
        IG = E[parent] - weighted_avg * E(children)

        E[x] = - np.sum(p(X_i) * no.log_2(P(x_i))) given that P(X) = #x/n
        """
        parent_entrophy = self._entropy(y)

        left_indexes, right_indexes = self._split(X=X_column, threshold=threshold)

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_indexes), len(right_indexes)
        w_l, w_r = n_l/n, n_r/n
        
        E_children = w_l* self._entropy(y[left_indexes]) + w_r * self._entropy(y[right_indexes])

        return parent_entrophy - E_children



    def _split(self, X: np.array, threshold):
        """
        Split the values  for left and right given an array and a threshold"""

        left_indexes = np.argwhere(X <= threshold).flatten()
        right_indexes = np.argwhere(X > threshold).flatten()

        return left_indexes, right_indexes

    def _entropy(self, y) -> float:

        # Frequencies of X
        #val_counts = list(Counter(y).values())
        # hist = np.bincount(y)
        # vals = hist[np.argwhere(hist > 0)]
        
        # len_y = len(y)
        # #ps = np.array(val_counts)/len_y
        # ps = vals /len_y
        # pslog = np.log2(ps)
        # E_x = np.sum(np.multiply(ps, pslog))
        # return -E_x
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p>0])



    def _most_common_label(self, labels):
        value_counts = Counter(labels)
        val =  value_counts.most_common(1)[0][0]
        return val

    def _traverse_tree(self, x, node: TreeNode):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
