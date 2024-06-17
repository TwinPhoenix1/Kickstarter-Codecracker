import pandas as pd
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def preprocess_data(df):
    df = df[df['state'].isin(['successful', 'failed'])]
    df = df.drop_duplicates(keep='last')

    df['category'] = df['category'].fillna(json.dumps({key: '' for key in df['category']
                                                        .dropna().apply(json.loads)
                                                        .iloc[0].keys()}))
    df_new = df['category'].apply(json.loads).apply(pd.Series)
    df_new = df_new.add_prefix('category_')
    df = df.drop("category", axis=1)
    df = pd.concat([df, df_new], axis=1)
    df['campaign_length'] = (df['deadline'] - df['launched_at']) / (60 * 60 * 24)
    df['buildup_length'] = (df['launched_at'] - df['created_at']) / (60 * 60 * 24)
    df['goal_usd'] = df['goal'] * df['fx_rate']
    for column_name in ['created_at', 'launched_at', 'deadline']:
        df[column_name] = pd.to_datetime(df[column_name], unit='s')
        df[column_name + '_weekday'] = df[column_name].dt.day_name()
        df[column_name + '_two_hour_chunk'] = np.floor(df[column_name].dt.hour / 2)
        df[column_name + '_two_hour_chunk'] = df[column_name + '_two_hour_chunk'].astype(int)
        df = df.drop(column_name, axis=1)
    df = df.drop(['creator', 'currency_symbol', 'currency_trailing_code', 'state_changed_at',
                  'category_parent_name', 'id', 'country_displayable_name', 'current_currency',
                  'photo', 'source_url', 'video', 'category_id', 'category_analytics_name',
                  'category_position', 'slug', 'usd_type', 'static_usd_rate', 'location',
                  'urls', 'usd_exchange_rate', 'category_slug', 'category_parent_id', 'profile',
                  'is_disliked', 'is_liked', 'is_starrable', 'spotlight', 'is_launched', 'goal',
                  'category_urls', 'category_color', 'disable_communication', 'country'], axis=1)
    df = df.drop(['name', 'blurb'], axis=1)
    df = df.drop('staff_pick', axis=1)
    df = df.drop(['percent_funded', 'pledged', 'fx_rate', 'backers_count', 'converted_pledged_amount',
                  'usd_pledged'], axis=1)
    df['prelaunch_activated'] = df['prelaunch_activated'].astype(int)
    df_fit = df.drop('state', axis=1)
    return df, df_fit

def encode_data(X_train, X_test):
    transformer = ColumnTransformer(transformers=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'),
                                                   ['category_name', 'currency', 'created_at_weekday',
                                                    'launched_at_weekday','deadline_weekday',
                                                    'created_at_two_hour_chunk', 'launched_at_two_hour_chunk',
                                                    'deadline_two_hour_chunk'])],
                                     remainder='passthrough')
    X_train = pd.DataFrame(transformer.fit_transform(X_train), columns=transformer.get_feature_names_out())
    X_test = pd.DataFrame(transformer.transform(X_test), columns=transformer.get_feature_names_out())
    return X_train, X_test, transformer

def split_data(df_encoded):
    random.seed(0)
    min_count = min(df_encoded['state'].value_counts())

    df_1 = df_encoded[df_encoded['state'] == 'successful'].sample(n=min_count)
    df_0 = df_encoded[df_encoded['state'] == 'failed'].sample(n=min_count)
    df = pd.concat([df_0, df_1])
    
    df['state'] = df['state'].map({'successful': 1, 'failed': 0})
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['state'])
    df_train, df_val = train_test_split(df_train, test_size=0.176, random_state=42, stratify=df_train['state'])
    y_train, y_test, y_val = df_train['state'], df_test['state'], df_val['state']

    X_train = df_train.drop('state', axis=1)
    X_test = df_test.drop('state', axis=1)
    X_val = df_val.drop('state', axis=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def scale_data(df_train, df_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(df_train)
    X_test = scaler.transform(df_test)
    
    return X_train, X_test, scaler


def custom_scorer(y_true, y_pred):
    return precision_score(y_true, y_pred) ** 2

def train_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.15],
        'gamma': [0.3, 0.4],
    }

    xgb = XGBClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    grid_search = GridSearchCV(xgb, param_grid, cv=skf, scoring='precision', verbose=10)
    grid_search.fit(X_train, y_train)

    print(f'Best hyperparameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    print("Precision score: " + str(precision_score(y_test, y_pred)))
    return model


def create_prediction_pipeline(transformer, scaler, model, df_fit):
    categorical_cols = ['category_name', 'currency', 'created_at_weekday', 
                        'launched_at_weekday', 'deadline_weekday', 'created_at_two_hour_chunk', 
                        'launched_at_two_hour_chunk', 'deadline_two_hour_chunk']
    numeric_cols = ['campaign_length', 'buildup_length', 'goal_usd', 'prelaunch_activated']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', transformer, categorical_cols),
            ('scaler', scaler, numeric_cols)
        ]
    )
    preprocessor.fit(pd.DataFrame(df_fit, columns=categorical_cols + numeric_cols))
    verifier = FunctionTransformer(verify_data_types)

    pipeline = Pipeline([
        ('verifier', verifier),
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def verify_data_types(X):
    expected_types = {
        'category_name': str,
        'currency': str,
        'created_at_weekday': str,
        'launched_at_weekday': str,
        'deadline_weekday': str,
        'created_at_two_hour_chunk': int,
        'launched_at_two_hour_chunk': int,
        'deadline_two_hour_chunk': int,
        'campaign_length': float,
        'buildup_length': float,
        'goal_usd': float,
        'prelaunch_activated': int
    }
    for col, expected_type in expected_types.items():
        if not all(isinstance(x, expected_type) for x in X[col]):
            raise ValueError(f"Column {col} has incorrect data type. Expected {expected_type}, got {X[col].dtype}")
    return X


