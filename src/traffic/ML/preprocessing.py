import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2

def preprocess_data(data):
    if data.empty:
        raise ValueError("O conjunto de dados está vazio.")

    # limpar colunas com valores idênticos
    data = data.loc[:, (data != data.iloc[0]).any()]

    # tratar valores infinitos
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    X = data.drop('label', axis=1)
    y = data['label']

    if 'time' in X.columns:
        X = X.drop(columns=['time'])

    # identificar colunas numéricas e categóricas
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    # valores ausentes (colunas numéricas)
    imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

    # normalização (colunas numéricas)
    scaler = MinMaxScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # aplicar One-Hot Encoding às colunas categóricas (melhor estratégio)
    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
    encoded_data = pd.DataFrame(encoder.fit_transform(X[categorical_columns]),
                               columns=encoder.get_feature_names_out(categorical_columns))

    X_processed = pd.concat([X[numeric_columns], encoded_data], axis=1)

    # Seleção de features
    selector = SelectKBest(chi2, k=min(10, X_processed.shape[1]))
    X_selected = selector.fit_transform(X_processed, y)

    return X_selected, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns