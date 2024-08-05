import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib


def load_taxifare_data():
    """
    Loads the taxifare data from a CSV file.

    Returns:
        pandas.DataFrame: The taxifare data loaded from the CSV file.
    """
    csv_path = os.path.join("Data", "train.csv")
    return pd.read_csv(csv_path)


def train_and_save_model():
    """
    Trains the linear regression model on the taxifare data and saves the model
    to a file.
    """
    taxifare = load_taxifare_data()

    # Preparing appropriate list for carrying out further analysis
    continuous_predictors = [
        'trip_duration', 'distance_traveled', 'fare', 'tip',
        'miscellaneous_fees'
    ]
    target = 'total_fare'
    continuous_features = [
        'trip_duration', 'distance_traveled', 'fare', 'tip',
        'miscellaneous_fees', 'total_fare'
    ]
    discrete_features = ['num_of_passengers', 'surge_applied']

    no_of_nan = []  # List to hold number of NaN
    nan_rows_index = []  # List to hold indexes of rows containing NaN value

    # Check if any of the features contain NaN value and print the row
    for feature in continuous_features:
        no_of_nan.append(taxifare[feature].isnull().sum())
        nan_rows_index.append(
            taxifare[taxifare[feature].isnull()].index.tolist()
        )

    nan_stat = pd.DataFrame(
        [no_of_nan, nan_rows_index], index=['Number of NaN', 'NaN Row Index']
    )
    nan_stat = nan_stat.transpose(copy=True)
    nan_stat.index = continuous_features

    # Dropping missing values and NaN
    taxifare_2 = taxifare.dropna(subset=continuous_features)

    # Filter negative values from continuous
    # features since from the description
    # of features the values cannot be negative
    taxifare_3 = taxifare_2[
        (taxifare_2[continuous_features] > 0).all(axis=1)
    ]

    # Identifying outliers present in each feature using Inter-Quartile Range
    # (IQR) approach and dropping the corresponding rows from the dataset

    UL = []  # List for upper limit
    LL = []  # List for lower limit
    NOO = []  # List for number of outliers
    OI = []  # List for indexes of outliers

    for feature in continuous_features:
        Q1 = taxifare_3[feature].quantile(0.25)
        Q3 = taxifare_3[feature].quantile(0.75)
        IQR = Q3 - Q1
        UL.append(Q3 + 1.5 * IQR)
        LL.append(Q1 - 1.5 * IQR)
        outliers = taxifare_3[feature] > UL[-1]
        NOO.append(outliers.sum())
        OI.append(outliers.index[outliers].to_numpy())

    for feature in continuous_features:
        outlier_indexes = OI[continuous_features.index(feature)]
        taxifare_3 = taxifare_3.drop(outlier_indexes, errors='ignore')

    r2, mse = LinearModel(
        0.8, 0.2, taxifare_3, target, continuous_predictors + discrete_features
    )
    print(f"R2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    model = LinearRegression().fit(
        taxifare_3[continuous_predictors + discrete_features],
        taxifare_3[target]
    )
    joblib.dump(model, 'Models/model.joblib')


def LinearModel(train_size, test_size, data, target_feature, features):

    df_train, df_test = train_test_split(
        data, train_size=train_size, test_size=test_size, random_state=100
    )

    y_train = df_train[target_feature]
    X_train = df_train[features]

    y_test = df_test[target_feature]
    X_test = df_test[features]

    lm = LinearRegression()
    model = lm.fit(X_train, y_train)

    # predict 'total_fare' of X_test
    y_pred = model.predict(X_test)

    # evaluate the model on test set
    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    train_and_save_model()
