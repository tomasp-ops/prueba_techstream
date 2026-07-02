"""
ejemplo_uso_lstm.py
--------------------
Ejemplo de uso de la LSTMDetector ya entrenada (model_lstm.pt) sobre 2
secuencias de servidor hardcodeadas (40 pasos temporales cada una), para
mostrar en el video como se usaria el modelo sobre el HISTORIAL completo de
un servidor nuevo.

Casos:
    - "estable": fluctua alrededor de un valor base fijo, sin tendencia,
      igual que los servidores sin fallo del dataset de entrenamiento
    - "degradacion": arranca en carga normal y sube de forma progresiva y
      acelerada (misma logica que generate_data_temporal.py), terminando en
      zona de riesgo en los ultimos pasos

Las secuencias se construyen con formulas fijas (sin aleatoriedad) para que
el ejemplo sea 100% reproducible sin depender de una semilla.

Pasos:
    1) Construir las 2 secuencias (40 pasos x 4 features)
    2) Escalarlas con el MISMO scaler_lstm.joblib usado en entrenamiento
    3) Pasarlas por el modelo en modo eval() + no_grad()
    4) Clasificar con umbral 0.5 e imprimir de forma legible
"""

import joblib
import numpy as np
import torch

from model_lstm import LSTMDetector

N_PASOS = 40
UMBRAL = 0.5


def construir_secuencia_estable():
    """Servidor ESTABLE: fluctua con una pequena ondulacion alrededor de un
    valor base fijo, sin ninguna tendencia de subida."""
    pasos = np.arange(N_PASOS)
    ondulacion = 3 * np.sin(pasos / 3)  # ripple determinista, no ruido aleatorio

    uso_cpu = 40 + ondulacion
    uso_memoria = 0.45 * uso_cpu + 0.55 * (50 + ondulacion)
    temperatura_cpu = 32 + (uso_cpu / 100) * 46 + 0.5 * ondulacion
    trafico_red = 200 + 20 * np.sin(pasos / 5)

    return np.stack([temperatura_cpu, uso_cpu, uso_memoria, trafico_red], axis=1)


def construir_secuencia_degradacion():
    """Servidor en DEGRADACION: arranca en carga normal (~30% CPU) y sube de
    forma progresiva y acelerada (curva progreso**1.4, igual que en
    generate_data_temporal.py) hasta terminar en zona de riesgo (~95% CPU)."""
    pasos = np.arange(N_PASOS)
    progreso = (pasos / (N_PASOS - 1)) ** 1.4
    ondulacion = 2 * np.sin(pasos / 3)  # ripple determinista, no linea recta perfecta

    uso_cpu = 30 + progreso * 65 + ondulacion
    uso_memoria = 0.45 * uso_cpu + 0.55 * (40 + progreso * 20 + ondulacion)
    temperatura_cpu = 32 + (uso_cpu / 100) * 46 + 0.5 * ondulacion
    trafico_red = 200 + 20 * np.sin(pasos / 5)

    return np.stack([temperatura_cpu, uso_cpu, uso_memoria, trafico_red], axis=1)


casos = {
    "Servidor ESTABLE": construir_secuencia_estable(),
    "Servidor EN DEGRADACION": construir_secuencia_degradacion(),
}

# ---------------------------------------------------------------------------
# Cargar scaler y modelo ya entrenados
# ---------------------------------------------------------------------------
scaler = joblib.load("scaler_lstm.joblib")

model = LSTMDetector()
model.load_state_dict(torch.load("model_lstm.pt"))
model.eval()

# ---------------------------------------------------------------------------
# Prediccion caso por caso
# ---------------------------------------------------------------------------
for nombre_caso, secuencia in casos.items():
    # Escalado: el scaler espera filas sueltas (pasos x features), asi que
    # aplanamos, escalamos, y volvemos a dar forma (1, 40, 4) para la LSTM.
    secuencia_escalada = scaler.transform(secuencia).reshape(1, N_PASOS, 4)
    secuencia_t = torch.FloatTensor(secuencia_escalada)

    with torch.no_grad():
        probabilidad = model(secuencia_t).item()  # ya en probabilidad (Sigmoid aplicada)

    prediccion = "TERMINA EN FALLO" if probabilidad >= UMBRAL else "NO FALLA"

    cpu_inicio, cpu_final = secuencia[0, 1], secuencia[-1, 1]
    temp_inicio, temp_final = secuencia[0, 0], secuencia[-1, 0]

    print(f"--- {nombre_caso} ---")
    print(f"  uso_cpu: paso 0 = {cpu_inicio:.1f}%  ->  paso 39 = {cpu_final:.1f}%")
    print(f"  temperatura_cpu: paso 0 = {temp_inicio:.1f}C  ->  paso 39 = {temp_final:.1f}C")
    print(f"  Probabilidad de fallo: {probabilidad:.2%}")
    print(f"  Prediccion (umbral {UMBRAL}): {prediccion}")
    print()
