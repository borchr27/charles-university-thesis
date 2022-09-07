
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_blobs
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import unlabeled_indices, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.visualization import plot_decision_boundary, plot_utilities

# Generate data set.
X, y_true = make_blobs(n_samples=200, centers=4, random_state=0)
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

# GaussianProcessClassifier needs initial training data otherwise a warning will
# be raised by SklearnClassifier. Therefore, the first 10 instances are used as
# training data.
y[:10] = y_true[:10]

# Create classifier and query strategy.
clf = SklearnClassifier(GaussianProcessClassifier(random_state=0),classes=np.unique(y_true), random_state=0)
qs = UncertaintySampling(method='entropy')

# Execute active learning cycle.
n_cycles = 20
for c in range(n_cycles):
    query_idx = qs.query(X=X, y=y, clf=clf)
    y[query_idx] = y_true[query_idx]

# Fit final classifier.
clf.fit(X, y)

# Visualize resulting classifier and current utilities.
bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
unlbld_idx = unlabeled_indices(y)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_title(f'Accuracy score: {clf.score(X,y_true)}', fontsize=15)
plot_utilities(qs, X=X, y=y, clf=clf, feature_bound=bound, ax=ax)
plot_decision_boundary(clf, feature_bound=bound, confidence=0.6)
plt.scatter(X[unlbld_idx,0], X[unlbld_idx,1], c='gray')
plt.scatter(X[:,0], X[:,1], c=y, cmap='jet')
plt.show()