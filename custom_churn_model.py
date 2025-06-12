
import numpy as np
from collections import Counter

# -------------------------
# ExtraTreeQuantileClassifier
# -------------------------
class ExtraTreeQuantileClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, n_quantiles=10, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_quantiles = n_quantiles
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def _get_feature_indices(self):
        if self.max_features is None:
            return np.arange(self.n_features_)
        elif self.max_features == 'sqrt':
            max_feats = max(1, int(np.sqrt(self.n_features_)))
        elif isinstance(self.max_features, int):
            max_feats = min(self.max_features, self.n_features_)
        else:
            raise ValueError("Invalid value for max_features.")
        return np.random.choice(self.n_features_, max_feats, replace=False)

    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        if (self.max_depth is not None and depth >= self.max_depth) or            num_samples < self.min_samples_split or len(set(y)) == 1:
            return self._leaf(y)

        feat_idxs = self._get_feature_indices()
        best_feat, best_thresh, best_loss = None, None, float('inf')

        for feat in feat_idxs:
            thresholds = self._quantile_thresholds(X[:, feat])
            for thresh in thresholds:
                y_left = y[X[:, feat] <= thresh]
                y_right = y[X[:, feat] > thresh]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                loss = self._log_loss_split(y_left, y_right)
                if loss < best_loss:
                    best_feat, best_thresh, best_loss = feat, thresh, loss

        if best_feat is None:
            return self._leaf(y)

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature': best_feat, 'threshold': best_thresh,
                'left': left_subtree, 'right': right_subtree}

    def _leaf(self, y):
        prob = np.bincount(y, minlength=self.n_classes_) / len(y)
        return {'leaf': True, 'prob': prob}

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        while 'leaf' not in tree:
            feature, threshold = tree['feature'], tree['threshold']
            if inputs[feature] <= threshold:
                tree = tree['left']
            else:
                tree = tree['right']
        return np.argmax(tree['prob'])

    def predict_proba(self, X):
        return np.array([self._predict_proba(inputs, self.tree) for inputs in X])

    def _predict_proba(self, inputs, tree):
        while 'leaf' not in tree:
            feature, threshold = tree['feature'], tree['threshold']
            if inputs[feature] <= threshold:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree['prob']

    def _quantile_thresholds(self, X_col):
        quantiles = np.linspace(0, 100, self.n_quantiles + 2)[1:-1]
        return np.percentile(X_col, quantiles)

    def _log_loss_split(self, y_left, y_right):
        def log_loss_part(y):
            counts = np.bincount(y, minlength=self.n_classes_)
            prob = counts / counts.sum()
            return -np.sum(prob * np.log(prob + 1e-9))
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * log_loss_part(y_left) + (len(y_right) / n) * log_loss_part(y_right)


# -------------------------
# ExtraTreesQuantileEnsemble
# -------------------------
class ExtraTreesQuantileEnsemble:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=10, n_quantiles=10, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_quantiles = n_quantiles
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_classes_ = len(set(y))
        self.trees = []

        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = ExtraTreeQuantileClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_quantiles=self.n_quantiles,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

    def predict_proba(self, X):
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probas, axis=0)


# -------------------------
# Wrapper: Classifier + Threshold
# -------------------------
class ChurnClassifierWithThreshold:
    def __init__(self, model, threshold=0.3):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
