import sklearn.metrics
from matplotlib import pyplot as plt
import numpy as np, json
import models

class Metric(object):
    def __init__(self, metric_type, threshold=0.5, average='binary', k=0.3):
        """Initialize metric object

        Args:
            metric_type ([str]): name of the metric to use
        """
        metric_mapping = {
            "f1": sklearn.metrics.f1_score,
            "precision": sklearn.metrics.precision_score,
            "recall": sklearn.metrics.recall_score,
            "FDR": FDR,
            "TPR": sklearn.metrics.recall_score
        }
        assert metric_type in metric_mapping.keys(), "metric doesn't exist"
        self.metric_fn = metric_mapping[metric_type]
        self.metric_type = metric_type
        self.threshold = threshold
        self.average = average
        self.k = k

    def evaluate(self, y_true, y_pred):
        """evaluate the prediction result by the metric at a threshold

        Args:
            y_true ([array]): ground truth label
            y_pred ([array]): predicted score in range (0,1)
            thres: a threshold to assign 0 and 1 for predicted score

        Returns:
            score by metric
        """
        assert len(y_true) == len(y_pred)
        y_pred = (y_pred >= self.threshold)
        return self.metric_fn(y_true, y_pred, average=self.average)

    def evaluate_top_k(self, y_true, y_pred):
        """evaluate the prediction result for top k proportion of the data
        (assign 1 to prediction at top k proportion)
        only support precision and recall for binary classification

        Args:
            y_true ([array]): [ground truth label]
            y_pred ([array]): [predicted score in range (0,1)]
            k (float, optional): [interested proportion of the data]. Defaults to 0.3.

        Returns:
            score by metric
        """

        assert len(y_true) == len(y_pred)
        order = np.argsort(y_pred)[::-1]
        n = int(self.k*len(y_pred))
        pred = np.zeros_like(y_pred)
        pred[order[:n]] = 1
        assert self.metric_type in ['precision', 'recall']
        if self.metric_type == 'precision':
            return np.sum(y_true[order[:n]])/n, pred
        elif self.metric_type == 'recall':
            return np.sum(y_true[order[:n]])/np.sum(y_true), pred
        
    def cross_tabs_at_k(self, X, y_pred, feature_indices):
        """evaluate the prediction result for top k proportion of the data
        (assign 1 to prediction at top k proportion)
        only support precision and recall for binary classification

        Args:
            y_true ([array]): [ground truth label]
            y_pred ([array]): [predicted score in range (0,1)]
            k (float, optional): [interested proportion of the data]. Defaults to 0.3.

        Returns:
            score by metric
        """

        assert len(X) == len(y_pred)
        order = np.argsort(y_pred)[::-1]
        n = int(self.k*len(y_pred))
        assert self.metric_type in ['precision', 'recall']
        
        top_x = X[order[:n]]
        bottom_x = X[order[n:]]
        
        top_vals = []
        bottom_vals = []
        
        for f_idx in feature_indices:
            top_vals.append(np.mean(top_x[:, f_idx]))
            bottom_vals.append(np.mean(bottom_x[:, f_idx]))
        
        return top_vals, bottom_vals

def FDR(y_true, y_pred, average):
    return 1-sklearn.metrics.precision_score(y_true, y_pred, average=average)
        

def plot_PR_k(y_true, y_pred, step=0.0002, file_path="prk.png"):
    """plot precision recall vs population curve

    Args:
        y_true ([array]): ground truth label
        y_pred ([array]): predicted score in range (0,1)
        step (float, optional): [step for evaluation the precision and recall]. Defaults to 0.01.
    """

    assert len(y_true) == len(y_pred)
    order = np.argsort(y_pred)[::-1]
    proportions = np.arange(start=step, stop=1+1e-8, step=step)
    precision, recall = [],[]
    for p in proportions:
        n = int(p*len(y_pred))
        p_score = np.sum(y_true[order[:n]])/n
        r_score = np.sum(y_true[order[:n]])/np.sum(y_true)
        precision.append(p_score)
        recall.append(r_score)
    plt.clf()
    plt.plot(proportions, precision, label="precision")
    plt.plot(proportions, recall, label="recall")
    plt.legend()
    plt.title("PR-k curve")
    plt.savefig(file_path)


def plot_score_distribution(y_pred, file_path="distribution.png"):
    """plot predicted score distribution

    Args:
        y_pred ([array]): predicted score in range (0,1)
    """
    plt.clf()
    plt.hist(y_pred, bins=100)
    plt.title("score distribution")
    plt.savefig(file_path)


def plot_feature_importance(model, feature_names, top_n=20, file_path="feature_importance.png"):
    assert isinstance(model, models.RFClassifier) or isinstance(model, models.DTClassifier)
    f_imps = model.model.feature_importances_
    order = np.argsort(f_imps)[::-1][:top_n]
    sorted_f_imps = f_imps[order]
    sorted_feature_names = [feature_names[i] for i in order]
    y_pos = np.arange(len(sorted_feature_names))

    fig, ax = plt.subplots()
    ax.barh(y_pos, sorted_f_imps, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature name')
    ax.set_title('Feature Importances')
    fig.savefig(file_path, bbox_inches='tight')
    return sorted_feature_names, order
    

if __name__ == "__main__":
    m = Metric('precision')
    import numpy as np
    y_true = np.array([0,0,1,1])
    y_pred = np.array([0.3,0.4,0.8,0.9])
    print(m.evaluate_top_k(y_true, y_pred))

    a = np.random.rand(10000)
    b = a>0.5
    plot_PR_k(b, a)