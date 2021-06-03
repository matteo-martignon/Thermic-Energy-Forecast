from sklearn.model_selection import learning_curve, validation_curve
import numpy as np
import matplotlib.pyplot as plt

def my_learning_curve(estimator, X_train, y_train, title=None, ylim=None):
    train_sizes, train_scores, test_scores = \
    learning_curve( 
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training examples')
#     plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    # plt.savefig('images/06_05.png', dpi=300)
    if title is not None:
        plt.title(title, fontsize=18)
    plt.show()
    return None

def my_validation_curve(estimator, X_train, y_train, param_range, title=None, ylim=[0.8, 1.0]):
    train_scores, test_scores = validation_curve(
                    estimator=estimator, 
                    X=X_train, 
                    y=y_train,
                    param_range=param_range,
                    cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='Training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='Validation accuracy')

    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')

    plt.grid()
#     plt.xscale('log')
    plt.legend(loc='lower right')
#     plt.xlabel('Parameter C')
#     plt.ylabel('Accuracy')
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    # plt.savefig('images/06_06.png', dpi=300)
    if title is not None:
        plt.title(title)
    plt.show()
    return None
