import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_data(data, label_column='label'):
    """
    Preprocessa dados do dataset de tráfego de rede
    
    Args:
        data: DataFrame com os dados
        label_column: nome da coluna de label (padrão: 'label')
    
    Returns:
        X: Features preprocessadas
        y: Labels
        imputer: Imputer treinado
        scaler: Scaler treinado
        encoder: Encoder treinado
        selector: Feature selector treinado
        numeric_columns: Lista de colunas numéricas
        categorical_columns: Lista de colunas categóricas
    """
    
    # Remove linhas com label faltando
    data = data.dropna(subset=[label_column])
    
    # Separa features e target
    y = data[label_column]
    
    # Colunas a ignorar (identificadores, timestamps, label)
    ignore_columns = [
        label_column, 'timestamp', 'flow_id', 'datapath_id',
        'ip_src', 'ip_dst', 'eth_src', 'eth_dst'
    ]
    
    # Remove colunas a ignorar
    feature_data = data.drop(columns=[col for col in ignore_columns if col in data.columns])
    
    # Substitui infinitos por NaN
    feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Define colunas numéricas e categóricas
    numeric_columns = [
        'flow_duration_sec', 'flow_duration_nsec', 
        'packet_count', 'byte_count',
        'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond',
        'idle_timeout', 'hard_timeout'
    ]
    
    categorical_columns = [
        'ip_proto', 'icmp_code', 'icmp_type', 'flags',
        'tp_src', 'tp_dst'
    ]
    
    # Filtra apenas colunas que existem no dataset
    numeric_columns = [col for col in numeric_columns if col in feature_data.columns]
    categorical_columns = [col for col in categorical_columns if col in feature_data.columns]
    
    print(f"Colunas numéricas ({len(numeric_columns)}): {numeric_columns}")
    print(f"Colunas categóricas ({len(categorical_columns)}): {categorical_columns}")
    
    # === PROCESSAMENTO NUMÉRICO ===
    # Preencher valores ausentes com a mediana
    imputer = SimpleImputer(strategy='median')
    numeric_data = pd.DataFrame(
        imputer.fit_transform(feature_data[numeric_columns]),
        columns=numeric_columns,
        index=feature_data.index
    )
    
    # Normalização
    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric_data),
        columns=numeric_columns,
        index=feature_data.index
    )
    
    # === PROCESSAMENTO CATEGÓRICO ===
    if categorical_columns:
        # Preencher valores ausentes com 'unknown'
        categorical_data = feature_data[categorical_columns].fillna('unknown')
        
        # Converter para string
        for col in categorical_columns:
            categorical_data[col] = categorical_data[col].astype(str)
        
        # One-Hot Encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = pd.DataFrame(
            encoder.fit_transform(categorical_data),
            columns=encoder.get_feature_names_out(categorical_columns),
            index=feature_data.index
        )
        
        # Combinar numérico e categórico
        X_combined = pd.concat([numeric_scaled, encoded], axis=1)
    else:
        encoder = None
        X_combined = numeric_scaled
    
    # === SELEÇÃO DE FEATURES ===
    # Seleciona as top K features mais relevantes
    k_features = min(50, X_combined.shape[1])  # Ajuste conforme necessário
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X_combined, y)
    
    print(f"\nFeatures originais: {X_combined.shape[1]}")
    print(f"Features selecionadas: {X_selected.shape[1]}")
    
    return X_selected, y, imputer, scaler, encoder, selector, numeric_columns, categorical_columns


def preprocess_predict_data(data, imputer, scaler, encoder, selector, numeric_columns, categorical_columns):
    """
    Preprocessa dados para predição usando componentes já treinados
    
    Args:
        data: DataFrame com dados para predição
        imputer: Imputer treinado
        scaler: Scaler treinado
        encoder: Encoder treinado
        selector: Feature selector treinado
        numeric_columns: Lista de colunas numéricas
        categorical_columns: Lista de colunas categóricas
    
    Returns:
        X_processed: Features preprocessadas prontas para predição
    """
    
    # Substitui infinitos por NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # === PROCESSAMENTO NUMÉRICO ===
    # Garante que todas as colunas numéricas existem
    for col in numeric_columns:
        if col not in data.columns:
            data[col] = 0
    
    numeric_data = data[numeric_columns].copy()
    
    # Preencher valores ausentes
    numeric_imputed = pd.DataFrame(
        imputer.transform(numeric_data),
        columns=numeric_columns,
        index=data.index
    )
    
    # Normalizar
    numeric_scaled = pd.DataFrame(
        scaler.transform(numeric_imputed),
        columns=numeric_columns,
        index=data.index
    )
    
    # === PROCESSAMENTO CATEGÓRICO ===
    if categorical_columns and encoder is not None:
        # Garante que todas as colunas categóricas existem
        for col in categorical_columns:
            if col not in data.columns:
                data[col] = 'unknown'
        
        categorical_data = data[categorical_columns].fillna('unknown')
        
        # Converter para string
        for col in categorical_columns:
            categorical_data[col] = categorical_data[col].astype(str)
        
        # Aplicar encoding
        encoded = pd.DataFrame(
            encoder.transform(categorical_data),
            columns=encoder.get_feature_names_out(categorical_columns),
            index=data.index
        )
        
        # Combinar
        X_combined = pd.concat([numeric_scaled, encoded], axis=1)
    else:
        X_combined = numeric_scaled
    
    # === SELEÇÃO DE FEATURES ===
    X_selected = selector.transform(X_combined)
    
    return X_selected