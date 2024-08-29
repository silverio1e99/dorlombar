import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Importação da base
spine = pd.read_csv('data/Dataset_spine.csv')

# Balanceamento das classes
sns.countplot(x=spine.Class_att)
plt.title("Balanceamento das Classes")
plt.show()

# Relação entre 2 variáveis e a classe
groups = spine.groupby("Class_att")
for name, group in groups:
    plt.scatter(group["Col6"], group["Col8"], label=name)
plt.xlabel("Col6")
plt.ylabel("Col8")
plt.legend()
plt.show()

# Histograma
plt.hist(spine.Col8, bins=50, alpha=0.5, label='Col8')
plt.hist(spine.Col1, bins=50, alpha=0.5, label='Col1')
plt.hist(spine.Col5, bins=50, alpha=0.5, label='Col5')
plt.legend()
plt.title('Distribuição para atributos Col1, Col1, e Col5')
plt.show()

# Machine Learning
seed = 10000  # semente para reprodução de resultados

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(spine.loc[:, spine.columns != 'Class_att'],
                                                    spine['Class_att'], test_size=0.2, random_state=seed)

# Treinamento do modelo
model = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)  # tente mudar parâmetro para evitar overfitting
model.fit(X_train, y_train)

# Visualização gráfica da árvore de decisão
plot_tree(model, feature_names=list(pd.DataFrame(X_train).columns.values),
          class_names=['Abnormal', 'Normal'], rounded=True, filled=True)
plt.savefig('model/spine_tree.png', dpi=600)
plt.show()

# métrica de treino e teste
print('Acurácia de treino:', model.score(X_train, y_train))
print('Acurácia de teste:', model.score(X_test, y_test))

# Predição
y_pred = model.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.title('Matriz de Confusão')
ax.xaxis.set_ticklabels(['Abnormal', 'Normal'])
ax.yaxis.set_ticklabels(['Abnormal', 'Normal'])
plt.show()

# Salvar modelo treinado dentro da pasta model
joblib.dump(model, 'model/spine_model.pkl')
print('Modelo salvo com sucesso!')
