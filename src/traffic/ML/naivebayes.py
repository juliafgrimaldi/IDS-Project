import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import logging
import os
import pickle
from .preprocessing import preprocess_data

def train_naive_bayes(file_path):
    data = pd.read_csv(file_path)

    if data.empty:
            raise ValueError("O arquivo de treinamento está vazio.")

    X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)

    # Aplicar SMOTE para balancear as classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    nb_model = GaussianNB()
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_nb_model = grid_search.best_estimator_

    # Avaliação do modelo
    y_pred = best_nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, output_dict=True))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    numeric_features = numeric_columns.tolist() if hasattr(numeric_columns, 'tolist') else list(numeric_columns)
    categorical_features = encoder.get_feature_names_out(categorical_columns).tolist()
    all_features = numeric_features + categorical_features
    
    # Get selected feature names
    selected_features = all_features[selector.get_support()]

    model_bundle = {
    'model': best_nb_model,
    'selector': selector,
    'encoder': encoder,
    'imputer': imputer,
    'scaler': scaler,
    'accuracy': accuracy,
    'numeric_columns': numeric_columns,
    'categorical_columns': categorical_columns,
    'selected_features': selected_features
}

    with open('nb_model_bundle.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
    return best_nb_model, selector, encoder, imputer, scaler, accuracy,  numeric_columns, categorical_columns

def predict_naive_bayes(model, selector, encoder, imputer, scaler, predict_file,  numeric_columns, categorical_columns):
    predict_flow_dataset = pd.read_csv(predict_file)

    if predict_flow_dataset.empty:
            raise ValueError("O arquivo de predição está vazio.")
    
    predict_flow_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Separar as colunas numéricas e categóricas
    #numeric_columns = predict_flow_dataset.select_dtypes(include=[np.number]).columns
    #categorical_columns = predict_flow_dataset.select_dtypes(exclude=[np.number]).columns

    #P Preencher valores ausentes nas colunas categóricas
    predict_flow_dataset[categorical_columns] = predict_flow_dataset[categorical_columns].fillna('unknown')

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

    predict_flow_dataset['prediction'] = y_pred
    
    ddos_flows = predict_flow_dataset[predict_flow_dataset['prediction'] == 1]

    return y_pred, ddos_flows
