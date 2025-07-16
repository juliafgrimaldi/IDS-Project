import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ======================
# 1. Dados (você pode colocar os seus)
# ======================
data = {
    "Algorithm": ["Random Forest", "Decision Tree", "SVM", "Naive Bayes", "K-NN"],
    "Accuracy Score": [97.63, 97.20, 87.64, 70.26, 97.71],
    "Precision Score": [98.00, 99.887308, 99.926878, 99.958176, 99.978583],
    "Recall": [98.00, 99.991853, 99.862943, 92.125467, 99.968003],
    "F1 Score": [98.00, 99.937049, 99.884082, 95.882121, 99.973292]
}

df = pd.DataFrame(data)

# ======================
# 2. Gráfico de barras agrupadas
# ======================
metrics = ["Accuracy Score", "Precision Score", "Recall", "F1 Score", "AUC Score"]
colors = ['#4F81BD', '#C0504D', '#9BBB59', '#F79646', '#8064A2']  # cores similares ao da imagem

x = np.arange(len(df["Algorithm"]))  # posições no eixo x
width = 0.15  # largura das barras

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, df[metric], width, label=metric, color=colors[i])

plt.xticks(x + width*2, df["Algorithm"])
plt.ylabel("Score (%)")
plt.ylim(85, 101)
plt.title("Resultados dos Diferentes Algoritmos Sobre o Dataset")
plt.legend()
plt.tight_layout()
plt.show()

# ======================
# 3. (Opcional) Salvar tabela em PNG
# ======================
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
tbl = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.savefig("tabela_resultados.png", dpi=300, bbox_inches="tight")
plt.show()
