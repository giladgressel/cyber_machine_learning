import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    """
    This is a custom modification of the code present here:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    scoring: string, callable or None, optional, default: None
        A string (see model evaluation documentation) or a scorer callable object / function
        with signature scorer(estimator, X, y).
    """

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    axes.grid(True)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")
    return fig


def plot_validation_curve(estimator, X, y, ylim=None, cv=None,
                          n_jobs=None, param_name=None, param_range=None, scoring=None):
    """
    referred from :
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    :param estimator: object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    :param X: array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :param ylim: tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    :param cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    :param n_jobs: int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param param_range: array-like, shape (n_values,)
    The values of the parameter that will be evaluated.
    :param param_name: string
    Name of the parameter that will be varied.
    :param scoring: string, callable or None, optional, default: None
        A string (see model evaluation documentation) or a scorer callable object / function
        with signature scorer(estimator, X, y).
    :return: fig


    """
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                 scoring=scoring, n_jobs=n_jobs, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    axes.grid(True)
    axes.set_title(f"Validation Curve with {estimator.__class__}")
    axes.set_xlabel(f"{param_name}")
    axes.set_ylabel(f"{scoring}")
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    lw = 2
    axes.plot(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
    axes.fill_between(param_range, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.2,
                      color="darkorange", lw=lw)
    axes.plot(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
    axes.fill_between(param_range, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.2,
                      color="navy", lw=lw)
    axes.legend(loc="best")
    return fig
