"""
evaluate_lstm.py
-----------------
Evalua la LSTMDetector ya entrenada (model_lstm.pt) sobre los 24 servidores
de test, usando exactamente el mismo split por servidor y la misma semilla
que en train_lstm.py (mismos indices de servidores en train/test, ya que
train_test_split con random_state=42 y stratify=y es 100% reproducible).

Genera:
    - accuracy, precision, recall, F1-score (impresos por consola)
    - matriz_confusion_lstm.png
    - comparacion train loss final vs test loss/accuracy, para valorar
      overfitting
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from model_lstm import LSTMDetector

SEED = 42
torch.manual_seed(SEED)

IMAGENES_DIR = "imagenes"
os.makedirs(IMAGENES_DIR, exist_ok=True)

FEATURES = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]

# Loss final de entrenamiento reportado por train_lstm.py (epoca 150/150),
# usado mas abajo solo como referencia para comparar con la loss de test.
LOSS_FINAL_TRAIN = 0.0290

# ---------------------------------------------------------------------------
# 1) Reconstruir EXACTAMENTE el mismo tensor (n_servidores, n_pasos,
#    n_features) + etiquetas, y el mismo split train/test por servidor que
#    en train_lstm.py (mismo orden de servidores, mismo test_size,
#    random_state y stratify -> mismos 24 servidores de test).
# ---------------------------------------------------------------------------
df = pd.read_csv("datos_servidores_temporal.csv")
df = df.sort_values(["id_servidor", "paso_temporal"])

ids_servidores = sorted(df["id_servidor"].unique())
N_SERVIDORES = len(ids_servidores)
N_PASOS = df["paso_temporal"].nunique()
N_FEATURES = len(FEATURES)

X = np.zeros((N_SERVIDORES, N_PASOS, N_FEATURES), dtype=np.float32)
y = np.zeros(N_SERVIDORES, dtype=np.float32)

for i, id_srv in enumerate(ids_servidores):
    bloque = df[df["id_servidor"] == id_srv].sort_values("paso_temporal")
    X[i] = bloque[FEATURES].values
    y[i] = bloque["fallo"].max()

indices_servidores = np.arange(N_SERVIDORES)
idx_train, idx_test, y_train, y_test = train_test_split(
    indices_servidores,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

X_test = X[idx_test]

print(f"Test: {len(idx_test)} servidores ({y_test.mean():.2%} con fallo)")
print()

# ---------------------------------------------------------------------------
# 2) Cargar el scaler ya ajustado en train_lstm.py (solo con train) y
#    escalar el test con el (sin volver a ajustarlo).
# ---------------------------------------------------------------------------
scaler = joblib.load("scaler_lstm.joblib")
X_test_scaled = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)

X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# ---------------------------------------------------------------------------
# 3) Reconstruir el modelo y cargar los pesos entrenados.
# ---------------------------------------------------------------------------
model = LSTMDetector()
model.load_state_dict(torch.load("model_lstm.pt"))

# ---------------------------------------------------------------------------
# 4) Inferencia en modo evaluacion, sin gradientes. Pedimos los LOGITS
#    (return_logits=True) para poder calcular la loss de test de forma
#    comparable a la de entrenamiento (BCEWithLogitsLoss tambien espera
#    logits), y aparte aplicamos Sigmoid manualmente para obtener las
#    probabilidades usadas en la clasificacion final.
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_logits_test = model(X_test_t, return_logits=True)
    y_prob_test = torch.sigmoid(y_logits_test)

# pos_weight identico al usado en entrenamiento, para que la loss de test
# sea comparable a la de train (mismo criterio, misma "vara de medir").
n_negativos_train = (y_train == 0).sum()
n_positivos_train = (y_train == 1).sum()
pos_weight = torch.tensor(n_negativos_train / n_positivos_train, dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_test = criterion(y_logits_test, y_test_t).item()

y_prob_np = y_prob_test.numpy().flatten()
y_test_np = y_test_t.numpy().flatten()

# ---------------------------------------------------------------------------
# 5) Umbral 0.5 -> clase 0/1
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
# 7) Matriz de confusion, etiquetada.
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
        ax.text(j, i, etiquetas[i, j], ha="center", va="center", color="black", fontsize=11)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Prediccion: 0 (no falla)", "Prediccion: 1 (falla)"])
ax.set_yticklabels(["Real: 0 (no falla)", "Real: 1 (falla)"])
ax.set_title("Matriz de confusion LSTM (test set, 24 servidores)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
ruta_matriz = os.path.join(IMAGENES_DIR, "matriz_confusion_lstm.png")
fig.savefig(ruta_matriz, dpi=150)
plt.close(fig)

print(f"Matriz de confusion guardada en {ruta_matriz}")
print()

# ---------------------------------------------------------------------------
# 8) Comparacion train vs test, para valorar overfitting
# ---------------------------------------------------------------------------
print("Comparacion train vs test (posible overfitting):")
print(f"- Loss final de entrenamiento (epoca 150/150): {LOSS_FINAL_TRAIN:.4f}")
print(f"- Loss en test (mismo criterio BCEWithLogitsLoss + pos_weight):  {loss_test:.4f}")
print(f"- Accuracy en test: {acc:.4f}")
print()
diferencia_loss = loss_test - LOSS_FINAL_TRAIN
print(
    f"- La loss de test ({loss_test:.4f}) es {diferencia_loss:.4f} puntos "
    f"{'mayor' if diferencia_loss > 0 else 'menor'} que la de train ({LOSS_FINAL_TRAIN:.4f})."
)
if diferencia_loss > 0.15:
    print(
        "  Esa diferencia es considerable: el modelo probablemente esta sobreajustando "
        "(overfitting) al set de entrenamiento, algo esperable dado que solo tiene 96 "
        "secuencias de train para una LSTM con bastantes parametros. Aun asi, si la "
        "accuracy de test es alta, el patron de degradacion vs. estabilidad puede ser "
        "lo bastante distinguible como para que el overfitting no afecte demasiado a la "
        "clasificacion final."
    )
else:
    print(
        "  Esa diferencia es moderada/pequena: no hay señales fuertes de overfitting, el "
        "modelo generaliza razonablemente bien de train a test."
    )
