import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_knn(file_path):
    data = pd.read_csv(file_path)

    # Limpar colunas com valores idênticos
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Tratar valores infinitos, substituindo por NaN e preenchendo com a média
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Preencher valores ausentes
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Normalizar os dados com minmax
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

    # Separar features e label
    X = data_scaled.drop('label', axis=1) 
    y = data_scaled['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Seleção de Atributos
    selector = SelectKBest(chi2, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_selected, y_train)

    # Avaliação do modelo
    y_pred_knn = knn_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"K-NN Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred_knn))

    return knn_model, selector, scaler

def predict_knn(model, selector, scaler, predict_file):
    predict_flow_dataset = pd.read_csv(predict_file)
    predict_flow_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Normalizar e aplicar seleção de atributos
    X_predict_flow = scaler.transform(predict_flow_dataset)
    X_predict_selected = selector.transform(X_predict_flow)

    return model.predict(X_predict_selected)
