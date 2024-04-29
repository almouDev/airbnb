import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Function to randomize the DataFrame
def randomize(df):
    np.random.seed(1)

    # Use np.random.permutation to randomize the row order
    randomized_indices = np.random.permutation(df.index)

    # Use the randomized indices to reorder the DataFrame rows
    df_randomized = df.loc[randomized_indices]

    return df_randomized

# Function to clean columns containing the signe dollar
def cleanup_price_column(df):
    for column in ["price","cleaning_fee","security_deposit"]:
        df[column] = df[column].str.replace(',', '')
        df[column] = df[column].str.replace('$', '')
        df[column] = df[column].astype(float)
    return df
   

# Function to delete columns
def delete_columns(df):
    df = df.drop([col for col in df.columns if col.startswith("host_")], axis=1)
    return df

#generate function used to transform non numeric columns and non ordinal columns to ordinal numeric columns
def generate_ordinal_columns(df,non_numeric_columns):
    for column in non_numeric_columns:
        value_counts = df[column].value_counts()
        df[column] = df[column].map(value_counts)/len(df)    
    return df




#get non numeric column
def get_non_numeric_columns(df):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    return non_numeric_columns

#normalization
def normalize(df):
    #test if df contains price column
    if "price" not in df.columns:
        df_normalized = df.copy()
        scaler = StandardScaler()
        df_normalized[df.columns] = scaler.fit_transform(df[df.columns])
        return df_normalized
    price=df["price"]
    df=df.drop(["price"],axis=1)
    df_normalized = df.copy()
    scaler = StandardScaler()
    df_normalized[df.columns] = scaler.fit_transform(df[df.columns])
    df_normalized["price"]=price
    return df_normalized


#remove extreme values
def remove_extreme_values(df):
    for column in df.columns:
        z_scores = zscore(df[column])
        df = df[(z_scores <= 3) & (z_scores >= -3)]
    return df


#feature selection
def feature_selection(x_train,y_train, methode,n_features=11):
    if methode == "SelectKBest":
        # Implement SelectKBest feature selection method
        selector = SelectKBest(score_func=f_regression,k=n_features)
        # Add your code here

    elif methode == "SelectPercentile":
        percentile = int((n_features/x_train.shape[1])*100)
        # Implement SelectPercentile feature selection method
        selector = SelectPercentile(score_func=f_regression,percentile=percentile)
        # Add your code here

    elif methode == "RFE":
        # Implement Recursive Feature Elimination (RFE) method

        selector = RFE(estimator=KNeighborsRegressor(),n_features_to_select=n_features)
        # Add your code here
    else:
        raise ValueError("Invalid feature selection method")

    # Return the selected features
    selected_features = selector.fit_transform(x_train, y_train)
    selected_features_names=selector.get_support(indices=True)
    return x_train.columns[selected_features_names].tolist()


# methode utilisée pour creer le modele
def get_knn_instance(n_neighbors=5, algorithm="auto"):
    # Create an instance of KNeighborsRegressor
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm= algorithm)
    return knn_regressor

# methode utilisee pour entrainer le modele
def train_knn_regressor( x_train, y_train,knn_regressor):
    return knn_regressor.fit(x_train, y_train)

# methode utilisee pour predire le prix d'un logement
def predict_price(test_df, knn_regressor,features):
    # Use the .predict() method of the trained model to estimate prices on the test dataset (test_df)
    predictions = knn_regressor.predict(test_df[features])

    return predictions
# methode utilisee pour calculer RMSE
def get_cross_validation_score(train_df_features, train_df_target, knn_regressor,cv):
    rmse_scores = cross_val_score(knn_regressor, train_df_features, train_df_target, scoring='neg_root_mean_squared_error', cv=cv)
    rmse = -rmse_scores.mean()
    return rmse


# methode utilisée pour trouver la meilleur hyperparametres
def doRandomSearch(train_df_features, train_df_target,knn_regressor):

    # Define the hyperparameter grid for KNN
    param_grid = {
        'n_neighbors':range(2,25),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    # Perform random search to find the best hyperparameters
    random_search = RandomizedSearchCV(estimator=knn_regressor, param_distributions=param_grid, scoring='neg_root_mean_squared_error', cv=5)
    random_search.fit(train_df_features, train_df_target)

    # View the best hyperparameters
    best_params = random_search.best_params_
    return best_params

# methode utilisée pour trouver la meilleur hyperparametres
def doGridSearch(train_df_features, train_df_target,knn_regressor):

    # Define the hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=knn_regressor, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
    grid_search.fit(train_df_features, train_df_target)

    # View the best hyperparameters
    best_params = grid_search.best_params_
    return best_params

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance

 # Remove columns containing non-numeric values
    #df = df.drop(["city", "state", "room_type"], axis=1)

    # Remove columns containing numerical but not ordinal values
    #df = df.drop(["longitude", "latitude", "zipcode"], axis=1)

    # Remove columns describing the host rather than the accommodation