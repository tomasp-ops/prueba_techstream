"""
train_lstm.py
-------------
Entrena la LSTMDetector (definida en model_lstm.py) para predecir si un
servidor termina fallando, a partir de su secuencia completa de 40 pasos
temporales. A diferencia del MLP (que clasificaba filas sueltas), aqui la
unidad de prediccion es EL SERVIDOR COMPLETO (su secuencia de 40 lecturas),
asi que el split train/test se hace por servidor, no por fila.

Pasos:
    1) Cargar el CSV en formato largo y reorganizarlo en un tensor
       (n_servidores, n_pasos, n_features) + una etiqueta por servidor
    2) Split train/test 80/20 por servidor completo, estratificado por esa
       etiqueta
    3) Escalar features (fit solo con los servidores de train)
    4) Entrenar con BCEWithLogitsLoss (pos_weight) + Adam, 150 epocas
    5) Guardar model_lstm.pt y el scaler
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_lstm import LSTMDetector

SEED = 42
torch.manual_seed(SEED)

FEATURES = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]

# ---------------------------------------------------------------------------
# 1) Cargar CSV (formato largo) y reorganizar en (n_servidores, n_pasos,
#    n_features). La etiqueta de cada servidor es 1 si fallo en algun punto
#    de su secuencia (recordemos que "fallo" solo se marca en el ultimo paso
#    de los servidores en degradacion), 0 si nunca fallo.
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
    y[i] = bloque["fallo"].max()  # 1 si fallo en algun paso de la secuencia

print(f"Datos reorganizados: X={X.shape}, y={y.shape}")
print(f"Servidores con fallo: {int(y.sum())} / {N_SERVIDORES}")
print()

# ---------------------------------------------------------------------------
# 2) Split train/test 80/20 POR SERVIDOR (no por fila), estratificado por
#    la etiqueta, para mantener la proporcion de servidores que fallan en
#    ambos conjuntos.
# ---------------------------------------------------------------------------
indices_servidores = np.arange(N_SERVIDORES)
idx_train, idx_test, y_train, y_test = train_test_split(
    indices_servidores,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

X_train = X[idx_train]
X_test = X[idx_test]

print(f"Train: {len(idx_train)} servidores ({y_train.mean():.2%} con fallo)")
print(f"Test:  {len(idx_test)} servidores ({y_test.mean():.2%} con fallo)")
print()

# ---------------------------------------------------------------------------
# 3) Escalado de features. Ajustamos (fit) el StandardScaler solo con los
#    servidores de train (aplanando servidor x paso a filas sueltas para que
#    el scaler vea todas las lecturas de train), y lo aplicamos igual a
#    train y test, para no filtrar informacion del test al entrenamiento.
# ---------------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, N_FEATURES))


def escalar_secuencias(bloque):
    """Aplica el scaler ya ajustado a un array (n_servidores, n_pasos,
    n_features), preservando su forma 3D."""
    forma_original = bloque.shape
    aplanado = bloque.reshape(-1, N_FEATURES)
    escalado = scaler.transform(aplanado)
    return escalado.reshape(forma_original)


X_train_scaled = escalar_secuencias(X_train)
X_test_scaled = escalar_secuencias(X_test)

# ---------------------------------------------------------------------------
# 4) Conversion a tensores de PyTorch.
# ---------------------------------------------------------------------------
X_train_t = torch.FloatTensor(X_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# ---------------------------------------------------------------------------
# Modelo, loss y optimizador.
#    Igual que en el MLP: BCEWithLogitsLoss con pos_weight (proporcion de
#    servidores sin fallo / con fallo en train), porque tambien aqui la
#    clase "fallo" es minoritaria (~20%). Adam como optimizador.
# ---------------------------------------------------------------------------
n_negativos = (y_train == 0).sum()
n_positivos = (y_train == 1).sum()
pos_weight = torch.tensor(n_negativos / n_positivos, dtype=torch.float32)
print(f"pos_weight (servidores sin fallo/con fallo en train): {pos_weight.item():.4f}")
print()

model = LSTMDetector()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------------------
# 5) Bucle de entrenamiento manual (150 epocas), igual estructura que
#    train.py: forward -> loss -> zero_grad -> backward -> step.
# ---------------------------------------------------------------------------
N_EPOCAS = 150

for epoca in range(1, N_EPOCAS + 1):
    model.train()

    y_logits = model(X_train_t, return_logits=True)
    loss = criterion(y_logits, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoca % 10 == 0:
        print(f"Epoca {epoca:3d}/{N_EPOCAS} - loss: {loss.item():.4f}")

print()

# ---------------------------------------------------------------------------
# Guardado del modelo entrenado y del scaler.
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "model_lstm.pt")
joblib.dump(scaler, "scaler_lstm.joblib")

print("Modelo guardado en model_lstm.pt")
print("Scaler guardado en scaler_lstm.joblib")
