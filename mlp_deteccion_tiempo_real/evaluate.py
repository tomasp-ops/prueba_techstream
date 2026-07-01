"""
evaluate.py
-----------
Evalua el MLP ya entrenado (model.pt) sobre el conjunto de TEST, usando el
mismo split en tres partes (train 60% / validation 20% / test 20%) y la
misma semilla que en train.py, para reconstruir exactamente las mismas
filas de test: un conjunto que no se ha usado ni para entrenar ni para
elegir el pos_weight (eso se hizo con validation en train.py).

Genera:
    - accuracy, precision, recall, F1-score (impresos por consola)
    - matriz_confusion.png (matriz de confusion con las 4 celdas etiquetadas)
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

IMAGENES_DIR = "imagenes"
os.makedirs(IMAGENES_DIR, exist_ok=True)

from model import MLP

SEED = 42
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1) Reconstruir EXACTAMENTE el mismo split de 3 partes que en train.py
#    (mismas features, mismos test_size, mismo random_state, mismo stratify,
#    mismo orden de los dos cortes) para llegar al mismo conjunto de test.
# ---------------------------------------------------------------------------
df = pd.read_csv("datos_servidores.csv")

FEATURES = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]
X = df[FEATURES].values
y = df["fallo"].values

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,  # 0.25 * 0.80 = 0.20 del total
    random_state=SEED,
    stratify=y_train_val,
)

print(f"Test: {len(X_test)} filas, {int(y_test.sum())} fallos ({y_test.mean():.2%})")
print()

# ---------------------------------------------------------------------------
# 2) Cargar el scaler ya ajustado en train.py y escalar el test con el
#    (NO se vuelve a ajustar el scaler aqui, solo transform).
# ---------------------------------------------------------------------------
scaler = joblib.load("scaler.joblib")
X_test_scaled = scaler.transform(X_test)

X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# ---------------------------------------------------------------------------
# 3) Reconstruir el modelo con la misma arquitectura y cargar los pesos
#    entrenados desde model.pt.
# ---------------------------------------------------------------------------
model = MLP()
model.load_state_dict(torch.load("model.pt"))

# ---------------------------------------------------------------------------
# 4) Inferencia en modo evaluacion (desactiva dropout/batchnorm si los
#    hubiera) y sin calcular gradientes (no_grad), ya que no vamos a
#    entrenar, solo a predecir.
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_prob = model(X_test_t)

y_prob_np = y_prob.numpy().flatten()
y_test_np = y_test_t.numpy().flatten()

# ---------------------------------------------------------------------------
# 5) Umbral de 0.5: probabilidad >= 0.5 -> clase 1 (fallo previsto)
# ---------------------------------------------------------------------------
UMBRAL = 0.5
y_pred_np = (y_prob_np >= UMBRAL).astype(int)

# ---------------------------------------------------------------------------
# 6) Metricas de clasificacion
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np, zero_division=0)
recall = recall_score(y_test_np, y_pred_np, zero_division=0)
f1 = f1_score(y_test_np, y_pred_np, zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print()

# ---------------------------------------------------------------------------
# 7) Matriz de confusion, etiquetada con las 4 celdas:
#    TN (verdaderos negativos) | FP (falsos positivos)
#    FN (falsos negativos)     | TP (verdaderos positivos)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test_np, y_pred_np, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("Matriz de confusion:")
print(cm)
print()

etiquetas = np.array(
    [
        [f"Verdaderos\nNegativos (TN)\n{tn}", f"Falsos\nPositivos (FP)\n{fp}"],
        [f"Falsos\nNegativos (FN)\n{fn}", f"Verdaderos\nPositivos (TP)\n{tp}"],
    ]
)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            etiquetas[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=11,
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Prediccion: 0 (sin fallo)", "Prediccion: 1 (fallo)"])
ax.set_yticklabels(["Real: 0 (sin fallo)", "Real: 1 (fallo)"])
ax.set_title("Matriz de confusion (test set)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
ruta_matriz = os.path.join(IMAGENES_DIR, "matriz_confusion.png")
fig.savefig(ruta_matriz, dpi=150)
plt.close(fig)

print(f"Matriz de confusion guardada en {ruta_matriz}")
print()

# ---------------------------------------------------------------------------
# 8) Interpretacion breve (para el video)
# ---------------------------------------------------------------------------
print("Interpretacion:")
print(
    f"- El modelo acierta el {acc:.1%} de los casos en total (accuracy), pero en un dataset "
    "desbalanceado (~14% de fallos) esta metrica por si sola puede ser enganosa: un modelo que "
    "siempre predijera 'sin fallo' ya tendria un accuracy alto."
)
print(
    f"- El recall ({recall:.1%}) es la metrica mas critica aqui: mide que porcentaje de los "
    "fallos REALES fue detectado por el modelo. En este problema, un falso negativo (un fallo "
    "real que el modelo no detecta) es mucho mas costoso que un falso positivo (una alarma "
    "de mas), porque significa un servidor que falla sin previo aviso."
)
print(
    f"- La precision ({precision:.1%}) indica que, de las veces que el modelo predice fallo, "
    "ese porcentaje realmente lo era; un valor mas bajo aqui es un costo aceptable "
    "(revisar una alarma que no era real) a cambio de no perder fallos reales."
)
print(
    f"- El F1-score ({f1:.1%}) resume el balance entre precision y recall en un solo numero, "
    "util para comparar modelos, pero para decidir si este modelo es aceptable en produccion "
    "hay que mirar el recall de cerca y, si es bajo, considerar bajar el umbral de decision "
    "(actualmente 0.5) para priorizar detectar mas fallos reales."
)
