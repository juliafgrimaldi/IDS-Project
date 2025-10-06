import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
from preprocessing import preprocess_data

def train_knn(file_path):
    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("O arquivo de treinamento está vazio.")

    X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)

    # Balancear as classes com SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    # Grid Search para hiperparâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    
    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_knn_model = grid_search.best_estimator_

    # Avaliação
    y_pred_knn = best_knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_knn)
    
    print(f"K-NN Accuracy: {accuracy * 100:.2f}%")
    print("\nMelhores hiperparâmetros:")
    print(grid_search.best_params_)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_knn))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_knn))

    # Salvar bundle
    model_bundle = {
        'model': best_knn_model,
        'selector': selector,
        'encoder': encoder,
        'imputer': imputer,
        'scaler': scaler,
        'accuracy': accuracy,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns
    }

    with open('models/knn_model_bundle.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)

    print("\n✓ Modelo KNN salvo em: models/knn_model_bundle.pkl")
    
    return best_knn_model, selector, encoder, imputer, scaler, accuracy, numeric_columns, categorical_columns

def predict_knn(model, selector, encoder, imputer, scaler, predict_file,  numeric_columns, categorical_columns):
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
    predict_flow_dataset[numeric_columns] = imputer.transform(predict_flow_dataset[numeric_columns].values)

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
