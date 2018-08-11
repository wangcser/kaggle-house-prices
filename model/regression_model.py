# model
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# data_set
from data_set.data_pre_process import import_data

# data_set tools
import numpy as np


class BaseModel:
    def __init__(self):
        # data
        self.train, self.test, self.y_train, self.test_ID = import_data()
        # validation args
        self.n_folds = 5

        """base model"""

        # LASSO Regression: This model may be very sensitive to outliers
        #                   So we need to made it more robust on them
        #                   For that we use the sk-learn's Robustscaler() method on pipeline
        self.lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

        # Elastic Net Regression
        self.ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

        # Kernel Ridge Regression
        self.KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        # Gradient Boosting Regression: use huber loss that makes it robust to outliers
        self.GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                                max_depth=4, max_features='sqrt',
                                                min_samples_leaf=15, min_samples_split=10,
                                                loss='huber', random_state=5)

        # XGBoost
        self.model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                          learning_rate=0.05, max_depth=3,
                                          min_child_weight=1.7817, n_estimators=2200,
                                          reg_alpha=0.4640, reg_lambda=0.8571,
                                          subsample=0.5213, silent=True,
                                          random_state=7, nthread=-1)

        # LightGBM
        """
        self.model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                           learning_rate=0.05, n_estimators=720,
                                           max_bin=55, bagging_fraction=0.8,
                                           bagging_freq=5, feature_fraction=0.2319,
                                           feature_fraction_seed=9, bagging_seed=9,
                                           min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
        """
        # LightGBM with less warning
        self.model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                           learning_rate=0.05, n_estimators=720,
                                           max_bin=55,

                                           feature_fraction_seed=9, bagging_seed=9,
                                           )

        # model list
        self.models = None

    def rmsle_cv(self, model):
        """
        define a cross validation strategy
        We use the cross_val_score function of Sk-learn.
        However this function has not a shuffle attribute,
        we add then one line of code, in order to shuffle the data set prior to cross-validation
        :param model: input single model
        :return: rmsle
        """
        kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(self.train.values)
        rmlse = np.sqrt(-cross_val_score(model, self.train.values, self.y_train,
                                         scoring="neg_mean_squared_error", cv=kf))

        return rmlse

    def base_model_score(self):
        """
        cal the base model score on cv set.
        :return: None
        """
        lasso_score = self.rmsle_cv(self.lasso)
        print("Lasso score: {:.4f} ({:.4f})".format(lasso_score.mean(), lasso_score.std()))

        elastic_net_score = self.rmsle_cv(self.ENet)
        print("ElasticNet score: {:.4f} ({:.4f})".format(elastic_net_score.mean(), elastic_net_score.std()))

        kernel_ridge_score = self.rmsle_cv(self.KRR)
        print("KernelRidge score: {:.4f} ({:.4f})".format(kernel_ridge_score.mean(), kernel_ridge_score.std()))

        gboost_score = self.rmsle_cv(self.GBoost)
        print("GradientBoosting score: {:.4f} ({:.4f})".format(gboost_score.mean(), gboost_score.std()))

        xgboost_score = self.rmsle_cv(self.model_xgb)
        print("XGBoost score: {:.4f} ({:.4f})".format(xgboost_score.mean(), xgboost_score.std()))

        lgbm_score = self.rmsle_cv(self.model_lgb)
        print("LGBM score: {:.4f} ({:.4f})" .format(lgbm_score.mean(), lgbm_score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Stacking models
    simplest stacking approach: averaging base models -> voting
    - We begin with this simple approach of averaging base models.
    - We build a new class to extend sk-learn with our model
      and also to leverage encapsulation and code reuse (inheritance)
    """
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        """
        we define clones of the original models to fit the data_set
        :param X:
        :param y:
        :return: average models itself.
        """
        self.models_ = [clone(x) for x in self.models]  # model list

        # train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        """
        Now we do the predictions for cloned models and average them
        :param X:
        :return:
        """
        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    less simple Stacking: adding a meta-model
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        """
        We again fit the data_set on clones of the original models
        :param X:
        :param y:
        :return:
        """
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        """
        Do the predictions of all base models on the test data_set and use the averaged predictions as
        meta-features for the final prediction which is done by the meta-model
        :param X:
        :return:
        """
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

    def rmsle(self, y, y_pred):
        """
        cal the rmsle on predict data.
        :param y_pred:
        :return:
        """
        return np.sqrt(mean_squared_error(y, y_pred))


def main():
    model = BaseModel()
    model.base_model_score()

if __name__ == '__main__':
    main()