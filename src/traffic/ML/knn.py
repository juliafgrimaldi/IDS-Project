import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

def train_knn(file_path):
    data = pd.read_csv(file_path)

    # Limpar colunas com valores idênticos
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Tratar valores infinitos, substituindo por NaN e preenchendo com a média
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

     # Separar features e label
    X = data.drop('label', axis=1) 
    y = data['label']

    # Identificar colunas numéricas e categóricas (não numéricas)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    # Preencher valores ausentes (colunas numéricas)
    imputer = SimpleImputer(strategy='mean')
    X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

    # Normalizar os dados com minmax (colunas numéricas)
    scaler = MinMaxScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Aplicar One-Hot Encoding às colunas categóricas (melhor estratégio)
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' para evitar redundancias
    data_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]),
                                columns=encoder.get_feature_names_out(categorical_columns))
    
    data_combined = pd.concat([X[numeric_columns], data_encoded], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(data_combined, y, test_size=0.3, random_state=42)

    # Seleção de Atributos
    selector = SelectKBest(chi2, k=min(10, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_selected, y_train)

    # Avaliação do modelo
    y_pred_knn = knn_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"K-NN Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred_knn))

    return knn_model, selector, encoder, imputer, scaler

def predict_knn(model, selector, encoder, imputer, scaler, predict_file):
    predict_flow_dataset = pd.read_csv(predict_file)
    predict_flow_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Separar as colunas numéricas e categóricas
    numeric_columns = predict_flow_dataset.select_dtypes(include=[np.number]).columns
    categorical_columns = predict_flow_dataset.select_dtypes(exclude=[np.number]).columns

    # Preencher valores ausentes nas colunas numéricas
    predict_flow_dataset[numeric_columns] = imputer.transform(predict_flow_dataset[numeric_columns])

    # Normalizar os dados numéricos com o scaler que foi treinado
    X_predict_scaled = pd.DataFrame(scaler.transform(predict_flow_dataset[numeric_columns]), columns=numeric_columns)

    # Aplicar One-Hot Encoding nas colunas categóricas
    encoded = pd.DataFrame(encoder.transform(predict_flow_dataset[categorical_columns]),
                                columns=encoder.get_feature_names_out(categorical_columns))
    
    X_predict_combined = pd.concat([X_predict_scaled, encoded], axis=1)

    # Seleção de atributos
    X_predict_selected = selector.transform(X_predict_combined)

    y_pred = model.predict(X_predict_selected)

    return y_pred
