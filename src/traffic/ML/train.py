import os
import sys
from train_knn import train_knn
from train_svm import train_svm
from train_decision_tree import train_decision_tree
from train_random_forest import train_random_forest

def main():
    print("="*60)
    print("TREINAMENTO DE MODELOS - IDS PARA DETECÇÃO DE DDOS")
    print("="*60)
    
    # Caminho do dataset
    dataset_path = "./backend/traffic_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"\n✗ ERRO: Dataset não encontrado em {dataset_path}")
        print("Por favor, verifique o caminho do arquivo.")
        return
    
    # Criar diretório de modelos
    os.makedirs('models', exist_ok=True)
    
    print(f"\n📊 Dataset: {dataset_path}")
    print("\n🔄 Iniciando treinamento dos modelos...\n")
    
    models_trained = []
    
    # Treinar KNN
    try:
        print("\n" + "="*60)
        print("1/4 - Treinando K-Nearest Neighbors (KNN)")
        print("="*60)
        train_knn(dataset_path)
        models_trained.append("KNN")
    except Exception as e:
        print(f"\n✗ Erro ao treinar KNN: {e}")
        import traceback
        traceback.print_exc()
    
    # Treinar Random Forest
    try:
        print("\n" + "="*60)
        print("2/4 - Treinando Random Forest")
        print("="*60)
        train_random_forest(dataset_path)
        models_trained.append("Random Forest")
    except Exception as e:
        print(f"\n✗ Erro ao treinar Random Forest: {e}")
        import traceback
        traceback.print_exc()
    
    # Treinar Decision Tree
    try:
        print("\n" + "="*60)
        print("3/4 - Treinando Decision Tree")
        print("="*60)
        train_decision_tree(dataset_path)
        models_trained.append("Decision Tree")
    except Exception as e:
        print(f"\n✗ Erro ao treinar Decision Tree: {e}")
        import traceback
        traceback.print_exc()
    
    # Treinar SVM
    try:
        print("\n" + "="*60)
        print("4/4 - Treinando Support Vector Machine (SVM)")
        print("="*60)
        train_svm(dataset_path)
        models_trained.append("SVM")
    except Exception as e:
        print(f"\n✗ Erro ao treinar SVM: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumo final
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"\n✓ {len(models_trained)}/4 modelos treinados com sucesso:")
    for model in models_trained:
        print(f"  • {model}")
    
    print("\n📁 Modelos salvos em: ./models/")
    print("  • knn_model_bundle.pkl")
    print("  • randomforest_model_bundle.pkl")
    print("  • dt_model_bundle.pkl")
    print("  • svm_model_bundle.pkl")
    
    print("\n🚀 Próximo passo: Execute o controller Ryu")
    print("   ryu-manager controller_api_dataset.py")

if __name__ == "__main__":
    main()