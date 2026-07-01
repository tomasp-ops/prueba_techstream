"""
train.py
--------
Entrena el MLP (definido en model.py) para clasificar fallos de servidor a
partir de las 4 features de telemetria. Bucle de entrenamiento manual (sin
Trainer/high-level API) para poder explicar cada paso en el video.

Pasos:
    1) Cargar CSV y separar X (features) / y (target = fallo)
    2) Split en TRES partes (train 60% / validation 20% / test 20%),
       estratificado en los tres, para poder elegir hiperparametros en
       validation sin tocar test hasta la evaluacion final
    3) Escalar features con StandardScaler (fit SOLO con train)
    4) Comparar pos_weight=6.08 (calculado) vs pos_weight=3.5 entrenando un
       modelo con cada uno y evaluandolos SOLO en validation
    5) Entrenar el modelo FINAL (200 epocas) con el pos_weight ganador,
       usando train, y guardar sus pesos + el scaler
    6) El conjunto de test queda intacto: se evalua en evaluate.py
"""

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import MLP

# Semilla fija para reproducibilidad del split y de la inicializacion de pesos
SEED = 42
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1) Cargar datos y separar features (X) / target (y)
# ---------------------------------------------------------------------------
df = pd.read_csv("datos_servidores.csv")

FEATURES = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]
X = df[FEATURES].values
y = df["fallo"].values

# ---------------------------------------------------------------------------
# 2) Split en TRES partes: 60% train / 20% validation / 20% test,
#    estratificado en ambos cortes para conservar la proporcion de fallos
#    en los tres conjuntos.
#    Se hace en dos pasos: primero se separa el 20% de test, y del 80%
#    restante se separa un 25% (= 20% del total) para validation, quedando
#    el otro 75% (= 60% del total) como train.
# ---------------------------------------------------------------------------
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

print("Tamano y casos positivos (fallos) por conjunto:")
print(f"Train:      {len(X_train)} filas, {int(y_train.sum())} fallos ({y_train.mean():.2%})")
print(f"Validation: {len(X_val)} filas, {int(y_val.sum())} fallos ({y_val.mean():.2%})")
print(f"Test:       {len(X_test)} filas, {int(y_test.sum())} fallos ({y_test.mean():.2%})")
print()

# ---------------------------------------------------------------------------
# 3) Escalado de features con StandardScaler.
#    Ajustamos (fit) el scaler SOLO con train. Test NO se transforma aqui:
#    se mantiene intacto hasta evaluate.py (unica vez que se usa).
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.FloatTensor(X_train_scaled)
X_val_t = torch.FloatTensor(X_val_scaled)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)

N_EPOCAS = 200


def entrenar_modelo(pos_weight_valor, seed=SEED, verbose=False):
    """Entrena un MLP desde cero (misma inicializacion via seed) sobre
    train, usando BCEWithLogitsLoss con el pos_weight indicado. Devuelve el
    modelo entrenado."""
    torch.manual_seed(seed)
    modelo = MLP()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_valor, dtype=torch.float32))
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

    for epoca in range(1, N_EPOCAS + 1):
        modelo.train()
        y_logits = modelo(X_train_t, return_logits=True)
        loss = criterion(y_logits, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoca % 10 == 0:
            print(f"Epoca {epoca:3d}/{N_EPOCAS} - loss: {loss.item():.4f}")

    return modelo


def evaluar_en_validation(modelo):
    """Evalua un modelo (umbral 0.5) sobre validation y devuelve sus
    metricas."""
    modelo.eval()
    with torch.no_grad():
        y_prob_val = modelo(X_val_t)  # ya en probabilidad (return_logits=False por defecto)
    y_pred_val = (y_prob_val.numpy().flatten() >= 0.5).astype(int)
    y_val_np = y_val_t.numpy().flatten()

    return {
        "precision": precision_score(y_val_np, y_pred_val, zero_division=0),
        "recall": recall_score(y_val_np, y_pred_val, zero_division=0),
        "f1": f1_score(y_val_np, y_pred_val, zero_division=0),
    }


# ---------------------------------------------------------------------------
# 4) Comparacion de pos_weight en VALIDATION (no en test).
#    pos_weight=6.08 (~proporcion negativos/positivos real en train) vs
#    pos_weight=3.5 (version mas "suave", probada anteriormente).
# ---------------------------------------------------------------------------
n_negativos = (y_train == 0).sum()
n_positivos = (y_train == 1).sum()
ratio_calculado = n_negativos / n_positivos
print(f"pos_weight calculado (negativos/positivos en train): {ratio_calculado:.4f}")
print()

candidatos_pos_weight = {
    "6.08 (calculado)": ratio_calculado,
    "3.5 (fijo)": 3.5,
}

resultados_validation = {}
for nombre, valor in candidatos_pos_weight.items():
    print(f"Entrenando con pos_weight={nombre}...")
    modelo_candidato = entrenar_modelo(valor)
    metricas = evaluar_en_validation(modelo_candidato)
    resultados_validation[nombre] = metricas
    print(
        f"  Validation -> precision: {metricas['precision']:.4f}, "
        f"recall: {metricas['recall']:.4f}, f1: {metricas['f1']:.4f}"
    )
print()

# Elegimos la configuracion ganadora por F1 en validation (mejor balance
# precision/recall); en caso de empate de F1 se preferiria el recall mas
# alto, dado que en este problema no detectar un fallo real es mas costoso
# que una falsa alarma.
nombre_ganador = max(resultados_validation, key=lambda n: resultados_validation[n]["f1"])
pos_weight_ganador = candidatos_pos_weight[nombre_ganador]
print(f"Configuracion ganadora (mejor F1 en validation): pos_weight={nombre_ganador}")
print()

# ---------------------------------------------------------------------------
# 5) Entrenamiento del modelo FINAL con el pos_weight ganador (200 epocas),
#    mostrando el loss cada 10 epocas.
# ---------------------------------------------------------------------------
print(f"Entrenando modelo final con pos_weight={pos_weight_ganador:.4f}...")
model = entrenar_modelo(pos_weight_ganador, verbose=True)
print()

# ---------------------------------------------------------------------------
# 6) Guardado del modelo final entrenado y del scaler (ajustado solo con
#    train). El test set NO se ha usado en ningun punto de este script.
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "model.pt")
joblib.dump(scaler, "scaler.joblib")

print("Modelo guardado en model.pt")
print("Scaler guardado en scaler.joblib")
