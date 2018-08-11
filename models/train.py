# data_set tools
import numpy as np
import pandas as pd

# models
from models.regression_model import BaseModel, AveragingModels, StackingAveragedModels


def main():
    # 1. base models cv-test.
    models = BaseModel()
    models.base_model_score()

    # 2. average models test.
    # We just average four models here ENet, GBoost, KRR and lasso.
    # Of course we could easily add more models in the mix.
    averaged_models = AveragingModels(models=(models.ENet, models.GBoost, models.KRR, models.lasso))
    score = models.rmsle_cv(averaged_models)
    print("Averaged base models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # 3. meta-average models test.
    # To make the two approaches comparable (by using the same number of models),
    # we just average ENet KRR and GBoost, then we add lasso as meta-models.

    stacked_averaged_models = StackingAveragedModels(base_models=(models.ENet, models.GBoost, models.KRR),
                                                     meta_model=models.lasso)
    score = models.rmsle_cv(stacked_averaged_models)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # 4. Ensemble Stacked Regressor, XGBoost and LightGBM
    # We add XGBoost and LightGBM to the StackedRegressor defined previously.

    # StackedRegressor
    stacked_averaged_models.fit(models.train.values, models.y_train)
    stacked_train_pred = stacked_averaged_models.predict(models.train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(models.test.values))
    print("stacked average models train rmsle:", stacked_averaged_models.rmsle(models.y_train, stacked_train_pred))

    # XGBoost
    models.model_xgb.fit(models.train, models.y_train)
    xgb_train_pred = models.model_xgb.predict(models.train)
    xgb_pred = np.expm1(models.model_xgb.predict(models.test))
    print("XGBoost train rmsle:", stacked_averaged_models.rmsle(models.y_train, xgb_train_pred))

    # LightGBM
    models.model_lgb.fit(models.train, models.y_train)
    lgb_train_pred = models.model_lgb.predict(models.train)
    lgb_pred = np.expm1(models.model_lgb.predict(models.test.values))
    print("LightGBM train rmsle:", stacked_averaged_models.rmsle(models.y_train, lgb_train_pred))

    # cal the ensemble train rmsle
    ensemble_train = stacked_train_pred * 0.8 + xgb_train_pred * 0.1 + lgb_train_pred * 0.1
    print("RMSLE score on train data_set:", stacked_averaged_models.rmsle(models.y_train, ensemble_train))

    # 5. Ensemble prediction
    ensemble = stacked_pred * 0.8 + xgb_pred * 0.1 + lgb_pred * 0.1

    # 6. submission
    sub = pd.DataFrame()
    sub['Id'] = models.test_ID
    sub['SalePrice'] = ensemble

    data_path = '/media/super/Dev Data/ml_data_set/Kaggle_house_price'
    sub.to_csv(data_path + '/result/submission.csv', index=False)


if __name__ == '__main__':
    main()

