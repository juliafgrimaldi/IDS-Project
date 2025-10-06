import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
from preprocessing import preprocess_data

def train_decision_tree(file_path):
    """
    Treina modelo Decision Tree para detecção de DDoS
    
    Args:
        file_path: Caminho do dataset CSV
    
    Returns:
        Componentes do modelo treinado
    """
    print("="*60)
    print("TREINANDO DECISION TREE")
    print("="*60)
    
    # Carrega dados
    data = pd.read_csv(file_path)
    
    if data.empty:
        raise ValueError("O arquivo de treinamento está vazio.")
    
    print(f"\nDataset carregado: {len(data)} registros")
    
    # Filtra flows válidos (mesmo critério do controller)
    data = data[
        (data['packet_count'] > 10) & 
        (data['byte_count'] > 1000) &
        (data['flow_duration_sec'] >= 1)
    ].copy()
    
    print(f"Após filtragem: {len(data)} registros")
    
    # Preprocessa dados
    print("\nPreprocessando dados...")
    X, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns = preprocess_data(data)
    
    # Verifica distribuição de classes
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nDistribuição de classes:")
    for label, count in zip(unique, counts):
        print(f"  Classe {label}: {count} ({count/len(y)*100:.2f}%)")
    
    # Aplicar SMOTE para balancear as classes
    print("\nAplicando SMOTE para balanceamento...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
    print(f"Após SMOTE:")
    for label, count in zip(unique_resampled, counts_resampled):
        print(f"  Classe {label}: {count} ({count/len(y_resampled)*100:.2f}%)")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    print(f"\nTreino: {len(X_train)} amostras")
    print(f"Teste:  {len(X_test)} amostras")
    
    # Treina modelo
    print("\nTreinando Decision Tree...")
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt_model.fit(X_train, y_train)
    
    # Avaliação
    print("\nAvaliando modelo...")
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}")
    
    print("\nRelatório de Classificação:")
    report = classification_report(y_test

def predict_decision_tree(model, selector, encoder, imputer, scaler, predict_file, numeric_columns, categorical_columns):
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
