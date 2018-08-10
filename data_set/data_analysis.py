import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy import stats
from scipy.stats import norm, skew
from subprocess import check_output

from sklearn.preprocessing import LabelEncoder


# ignore annoying warning from sklearn and seaborn
def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn


def import_data():
    # check the files available in the directory.
    data_path = '/media/super/Dev Data/ml_data_set/Kaggle_house_price'
    # print(check_output(['ls', data_path]).decode('utf-8'))

    # import dataset as pandas dataframe
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    # print(train_df.head(5))

    # check the numbers of samples and features
    # train_df is 1460*81, test_df is 1459*80
    # print("The train data_set size before dropping Id feature is : {} ".format(train_df.shape))
    # print("The test data_set size before dropping Id feature is : {} ".format(test_df.shape))

    # save the ID column
    train_ID = train_df['Id']
    test_ID = test_df['Id']
    # drop the Id column, since it's unnecessary for the prediction.
    train_df.drop('Id', axis=1, inplace=True)
    test_df.drop('Id', axis=1, inplace=True)

    # check again the data_set size after dropping the 'Id' variable
    # print("\nThe train data_set size after dropping Id feature is : {} ".format(train_df.shape))
    # print("The test data_set size after dropping Id feature is : {} ".format(test_df.shape))

    # Data Processing

    # analysis the outliers
    # fig, ax = plt.subplots()
    # ax.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'])
    # plt.xlabel('GrLivArea', fontsize=12)
    # plt.ylabel('SalePrice', fontsize=12)
    # plt.show()

    # We can see at the bottom right two with extremely large GrLivArea that are of a low price.
    # These values are huge outliers. Therefore, we can safely delete them.
    train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

    # check the gragh again.
    # fig, ax = plt.subplots()
    # ax.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'])
    # plt.xlabel('GrLivArea', fontsize=12)
    # plt.ylabel('SalePrice', fontsize=12)
    # plt.show()


    # target vars.

    """
    # analyze the SalePrice.
    sns.distplot(train_df['SalePrice'], fit=norm)
    
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_df['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    
    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    
    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(train_df['SalePrice'], plot=plt)
    plt.show()
    """
    # The target variable is right skewed. As (linear) models love normally distributed data_set ,
    # we need to transform this variable and make it more normally distributed.

    # normalization.
    # We use the numpy function log1p which  applies log(1+x) to all elements of the column
    train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

    # Check the new distribution
    # sns.distplot(train_df['SalePrice'], fit=norm)

    # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(train_df['SalePrice'])
    # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    # plt.ylabel('Frequency')
    # plt.title('SalePrice distribution')
    #
    # # Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(train_df['SalePrice'], plot=plt)
    # plt.show()

    # feature engineering.

    # let's first concatenate the train and test data_set in the same dataFrame.

    # drop the SalePrice label
    ntrain = train_df.shape[0]
    ntest = test_df.shape[0]
    y_train = train_df.SalePrice.values
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    # print("all_data size is : {}".format(all_data.shape))

    """
    # analyze the missing data_set with missing ratio
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
    print(missing_data.head(20))
    
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data_set by feature', fontsize=15)
    plt.show()
    """

    """
    # data_set correlation analysis
    # Correlation map to see how features are correlated with SalePrice
    corrmat = train_df.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    """

    # add the missing data_set

    # PoolQC : data_set description says NA means "No Pool".
    # That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

    # MiscFeature : data_set description says NA means "no misc feature"
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

    # Alley : data_set description says NA means "no alley access"
    all_data["Alley"] = all_data["Alley"].fillna("None")

    # Fence : data_set description says NA means "no fence"
    all_data["Fence"] = all_data["Fence"].fillna("None")

    # FireplaceQu : data_set description says NA means "no fireplace"
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    # LotFrontage : Since the area of each street connected to the house property most likely
    # have a similar area to other houses in its neighborhood,
    # we can fill in missing values by the median LotFrontage of
    # the neighborhood.
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data_set with None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')

    # GarageYrBlt, GarageArea and GarageCars : Replacing missing data_set with 0 (Since No garage = no cars in such garage.)
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath:
    # missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2:
    # For all these categorical basement-related features, NaN means that there is no basement.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')

    # MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses.
    # We can fill 0 for the area and None for the type.
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    # MSZoning (The general zoning classification) :
    # 'RL' is by far the most common value. So we can fill in missing values with 'RL'
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    # Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA.
    # Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling.
    # We can then safely remove it.
    all_data = all_data.drop(['Utilities'], axis=1)

    # Functional : data_set description says NA means typical
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    # KitchenQual: Only one NA value, and same as Electrical,
    # we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value.
    # We will just substitute in the most common string
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

    # SaleType : Fill in again with most frequent which is "WD"
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    # MSSubClass : Na most likely means No building class. We can replace missing values with None
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    """
    # Is there any remaining missing value?
    # Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print(missing_data.head())
    """

    # more feature engineering
    # Transforming some numerical variables that are really categorical

    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    # Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    # Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # Label Encoding some categorical variables that may contain information in their ordering set

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape
    # print('Shape all_data: {}'.format(all_data.shape))

    # Adding one more important feature

    # Adding total sq-footage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # skewed features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    # print(skewness.head(10))

    # Box Cox Transformation of (highly) skewed features
    skewness = skewness[abs(skewness) > 0.75]
    # print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # all_data[skewed_features] = np.log1p(all_data[skewed_features])

    # Getting dummy categorical features
    all_data = pd.get_dummies(all_data)
    # print(all_data.shape)

    # Getting the new train and test sets.
    train_df = all_data[:ntrain]
    test_df = all_data[ntrain:]

    return train_df, test_df, y_train, test_ID
