"""Reusable helpers for the SVM & Kernel Trick project.

Everything that is used more than once in the notebook lives here:
data generation, pipeline construction, hyper-parameter search,
evaluation and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# Colours used consistently across every figure
CLASS_COLORS = ["#2b6cb0", "#dd6b20"]          # class 0 = blue, class 1 = orange
REGION_CMAP = ListedColormap(["#c3dafe", "#feebc8"])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_moons(n_samples=600, noise=0.25, random_state=RANDOM_STATE):
    """Two interleaving half-moons: a classic non-linearly separable dataset."""
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def plot_dataset(X, y, ax=None, title=None, alpha=0.8, s=25, marker="o", label_prefix=""):
    """Scatter plot of a 2-D binary dataset."""
    ax = ax or plt.gca()
    for cls in (0, 1):
        ax.scatter(
            X[y == cls, 0], X[y == cls, 1],
            c=CLASS_COLORS[cls], s=s, alpha=alpha, marker=marker,
            edgecolors="white", linewidths=0.4,
            label=f"{label_prefix}class {cls}",
        )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_pipeline(kernel="rbf", **svc_params):
    """StandardScaler -> SVC.

    Wrapping the scaler in the pipeline guarantees that, during cross-validation,
    the scaler is fitted on the training folds only (no data leakage).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, **svc_params)),
    ])


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


def grid_search(pipeline, param_grid, X_train, y_train, n_splits=5, refit="balanced_accuracy"):
    """Stratified k-fold grid search using several metrics; refits on `refit`."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring=SCORING, refit=refit,
        return_train_score=True, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def print_cv_table(search, params, top=None, sort_by="balanced_accuracy"):
    """Print the CV results for the given parameter names as a plain-text table."""
    res = search.cv_results_
    order = np.argsort(res[f"rank_test_{sort_by}"])
    if top:
        order = order[:top]
    header = "".join(f"{p.replace('svc__', ''):>8}" for p in params)
    header += f"{'train_bal':>11}{'cv_bal':>9}{'cv_std':>8}{'cv_acc':>8}{'cv_f1':>8}"
    print(header)
    print("-" * len(header))
    for i in order:
        row = "".join(f"{str(res['param_' + p][i]):>8}" for p in params)
        row += (f"{res['mean_train_' + sort_by][i]:>11.3f}"
                f"{res['mean_test_' + sort_by][i]:>9.3f}"
                f"{res['std_test_' + sort_by][i]:>8.3f}"
                f"{res['mean_test_accuracy'][i]:>8.3f}"
                f"{res['mean_test_f1'][i]:>8.3f}")
        print(row)


def evaluate(model, X_test, y_test):
    """Test-set metrics for a fitted pipeline."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "n_support_vectors": int(model.named_steps["svc"].n_support_.sum()),
    }


def print_summary(results):
    """results: dict name -> dict(best_params, cv_score, metrics)."""
    header = (f"{'model':<12}{'best parameters':<38}{'CV bal.acc':>11}"
              f"{'test acc':>10}{'test bal.acc':>13}{'test F1':>9}{'#SV':>6}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["metrics"]
        params = ", ".join(f"{k.replace('svc__', '')}={v}" for k, v in r["best_params"].items())
        print(f"{name:<12}{params:<38}{r['cv_score']:>11.3f}"
              f"{m['accuracy']:>10.3f}{m['balanced_accuracy']:>13.3f}"
              f"{m['f1']:>9.3f}{m['n_support_vectors']:>6}")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _grid(X, pad=0.6, resolution=300):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    return xx, yy


def plot_decision_boundary(model, X_train, y_train, X_test=None, y_test=None,
                           ax=None, title=None, show_sv=True, show_margin=True):
    """Draw predicted regions, the decision boundary (f(x)=0), the margins
    (f(x)=±1) and circle the support vectors.

    The model is a fitted Pipeline, so the grid is given in the ORIGINAL feature
    space and the pipeline scales it internally.
    """
    ax = ax or plt.gca()
    X_all = X_train if X_test is None else np.vstack([X_train, X_test])
    xx, yy = _grid(X_all)
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z > 0, cmap=REGION_CMAP, alpha=0.9)
    ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
    if show_margin:
        ax.contour(xx, yy, Z, levels=[-1, 1], colors="k", linewidths=1, linestyles="--")

    plot_dataset(X_train, y_train, ax=ax, s=18, alpha=0.75, label_prefix="train ")
    if X_test is not None:
        plot_dataset(X_test, y_test, ax=ax, s=40, alpha=1.0, marker="^", label_prefix="test ")

    if show_sv:
        sv_idx = model.named_steps["svc"].support_
        sv = X_train[sv_idx]  # support_ indexes the training rows (original scale)
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="k",
                   linewidths=0.9, label=f"support vectors ({len(sv_idx)})")

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_confusion(cm, ax=None, title=None):
    ax = ax or plt.gca()
    ConfusionMatrixDisplay(cm, display_labels=["class 0", "class 1"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_cv_heatmap(search, row_param, col_param, ax=None, metric="balanced_accuracy", title=None):
    """Heat-map of mean CV score over a 2-D parameter grid."""
    ax = ax or plt.gca()
    res = search.cv_results_
    rows = sorted(set(res["param_" + row_param]))
    cols = sorted(set(res["param_" + col_param]))
    M = np.full((len(rows), len(cols)), np.nan)
    for r, c, s in zip(res["param_" + row_param], res["param_" + col_param],
                       res["mean_test_" + metric]):
        M[rows.index(r), cols.index(c)] = s
    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(cols)), [str(c) for c in cols])
    ax.set_yticks(range(len(rows)), [str(r) for r in rows])
    ax.set_xlabel(col_param.replace("svc__", ""))
    ax.set_ylabel(row_param.replace("svc__", ""))
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < np.nanmean(M) else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label=f"CV {metric.replace('_', ' ')}")
    if title:
        ax.set_title(title, fontsize=10)
    return ax
