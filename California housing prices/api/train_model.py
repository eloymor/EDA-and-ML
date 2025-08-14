import os
import datetime
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import optuna
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

TARGET = "median_house_value"

engine = create_engine('sqlite:///../etl_data.db')
Base = declarative_base()


class Data(Base):
    __tablename__ = 'processed_data'
    index = Column(Integer, primary_key=True)
    housing_median_age = Column(Integer)
    total_rooms = Column(Integer)
    total_bedrooms = Column(Integer)
    population = Column(Integer)
    households = Column(Integer)
    median_income = Column(Float)
    median_house_value = Column(Float)
    ocean_proximity = Column(Integer)
    clusters = Column(Integer)


Session = sessionmaker(bind=engine)


def load_data(db_name, table_name)-> pd.DataFrame:

    session = Session()
    data_dicts = []
    try:
        data = session.query(Data).all()
        for item in data:
            data_dict = {column.name: getattr(item, column.name) for column in item.__table__.columns}
            data_dict.pop("index", None)
            data_dicts.append(data_dict)
    except Exception as e:
        print(f"Error retrieving data: {e}")
    finally:
        session.close()
        df = pd.DataFrame(data_dicts)

    return df

def create_dmatrix_sets(df: pd.DataFrame):

    df_train = df.sample(frac=0.7)
    df_test = df.drop(df_train.index)

    print(f"Training set size: {len(df_train)}\nTest set size: {len(df_test)}")

    X_train = df_train
    Y_train = df_train.pop(TARGET)
    X_test = df_test
    Y_test = df_test.pop(TARGET)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    return dtrain, dtest

def objective(trial, dtrain, dtest):
    """
    Function to optimize hyperparameters using Optuna.
    :param trial: Used by Optuna to optimize hyperparameters.
    :return: List of RMSE scores.
    """
    num_round = 200
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    param = {
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "objective": "reg:squaredlogerror",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(params=param,
                    dtrain=dtrain,
                    num_boost_round=num_round,
                    evals=evallist,
                    early_stopping_rounds=10,
                    verbose_eval=False)

    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    return root_mean_squared_error(ypred, dtest.get_label())


def find_best_params(dtrain, dtest, n_trials=10):
    """
    Optimize hyperparameters using Optuna and return the best parameters.
    The function executes hyperparameter optimization for a specified number
    of trials using the Optuna optimization framework. It returns a dictionary
    of the best parameters found during the optimization process, with certain
    fixed parameters added after the optimization.

    :param n_trials: The number of trials to perform during optimization.
    :type n_trials: int
    :return: A dictionary containing the optimal hyperparameters,
        including fixed predefined options for tree method, device,
        and the objective function.
    :rtype: dict
    """
    study = optuna.create_study(direction="minimize")
    start_time = datetime.datetime.now()
    print("Starting hyperparameter optimization...")
    study.optimize(lambda trial: objective(trial, dtrain, dtest), n_trials=n_trials)
    end_time = datetime.datetime.now()
    print(f"Optimization completed in {(end_time - start_time).total_seconds()} seconds!")

    best_params = study.best_params
    best_params["tree_method"] = "hist"
    best_params["device"] = "cuda"
    best_params["objective"] = "reg:squaredlogerror"

    return best_params


def train_model(dtrain, dtest, params):
    """
    Trains a model using the provided training and testing datasets along with the specified parameters.

    This function initializes and trains a model using the XGBoost library. It evaluates the model's
    performance on both the training and testing datasets during the training process. The training
    stops early if the evaluation metric fails to improve over a given number of iterations.

    :param dtrain: DMatrix containing the training data.
    :param dtest: DMatrix containing the testing/evaluation data.
    :param params: Dictionary of parameters for the XGBoost model.
    :return: Trained XGBoost Booster model.
    """
    print("Training model...")
    start_time = datetime.datetime.now()
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round=200,
                    evals=evallist,
                    early_stopping_rounds=10,
                    verbose_eval=False)
    end_time = datetime.datetime.now()
    print(f"Model training completed in {(end_time - start_time).total_seconds()} seconds!")

    return bst

# For API training
async def api_train(db_name: str = "etl_data.db", table_name: str = "processed_data"):
    db_name = os.path.join("..", db_name)
    df = load_data(db_name, table_name)
    print(f"Successfully loaded data from: {db_name}")
    print(f"Shape of the DataFrame: {df.shape}")

    dtrain, dtest = create_dmatrix_sets(df)

    best_params = find_best_params(dtrain, dtest, n_trials=10)
    bst = train_model(dtrain, dtest, best_params)

    model_path = os.path.join("..", "models", "xgboost_model.json")
    if not os.path.exists(os.path.join("..", "models")):
        os.makedirs(os.path.join("..", "models"), exist_ok=True)
    bst.save_model(model_path)

    pass


# For CLI training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on processed data.")
    parser.add_argument("--db_name", type=str, default="etl_data.db",
                        help="The name of the SQLite database file.")
    parser.add_argument("--table_name", type=str, default="processed_data",
                        help="The name of the table to store the processed data.")
    args = parser.parse_args()

    df = load_data(args.db_name, args.table_name)
    print(f"Successfully loaded data from: {args.db_name}")
    print(f"Shape of the DataFrame: {df.shape}")

    dtrain, dtest = create_dmatrix_sets(df)

    best_params = find_best_params(dtrain, dtest, n_trials=10)
    bst = train_model(dtrain, dtest, best_params)

    model_path = os.path.join("..", "models", "xgboost_model.json")
    if not os.path.exists(os.path.join("..", "models")):
        os.makedirs(os.path.join("..", "models"), exist_ok=True)
    bst.save_model(model_path)
    print(f"Model saved successfully to {model_path}")
