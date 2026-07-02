"""
ejemplo_uso.py
--------------
Ejemplo de uso del MLP ya entrenado (model.pt) sobre 3 casos de servidor
hardcodeados, para mostrar en el video como se usaria el modelo en la
practica sobre un snapshot nuevo (no visto durante el entrenamiento).

Pasos por cada caso:
    1) Definir los 4 valores de entrada (temperatura_cpu, uso_cpu,
       uso_memoria, trafico_red)
    2) Escalarlos con el MISMO scaler.joblib usado en entrenamiento (nunca
       se reajusta el scaler sobre datos nuevos, solo transform)
    3) Pasarlos por el modelo en modo eval() + no_grad() para obtener la
       probabilidad de fallo
    4) Clasificar con el umbral 0.5 e imprimir todo de forma legible
"""

import joblib
import torch

from model import MLP

FEATURES = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]
UMBRAL = 0.5

# ---------------------------------------------------------------------------
# Casos de ejemplo hardcodeados (para explicar en el video):
#   - "sano": todas las metricas en rango normal, sin ninguna senal de riesgo
#   - "en_riesgo": temperatura, CPU y memoria altas a la vez (varias zonas de
#     riesgo simultaneas, el patron que el modelo aprendio a asociar a fallo)
#   - "ambiguo": solo un par de metricas elevadas, el resto normal (zona
#     limite, donde el modelo tiene que "decidir" con menos senales claras)
# ---------------------------------------------------------------------------
casos = {
    "Servidor SANO": {
        "temperatura_cpu": 50.0,
        "uso_cpu": 40.0,
        "uso_memoria": 45.0,
        "trafico_red": 200.0,
    },
    "Servidor EN RIESGO": {
        "temperatura_cpu": 85.0,
        "uso_cpu": 92.0,
        "uso_memoria": 88.0,
        "trafico_red": 800.0,
    },
    "Servidor AMBIGUO": {
        "temperatura_cpu": 72.0,
        "uso_cpu": 78.0,
        "uso_memoria": 55.0,
        "trafico_red": 300.0,
    },
}

# ---------------------------------------------------------------------------
# Cargar scaler y modelo ya entrenados
# ---------------------------------------------------------------------------
scaler = joblib.load("scaler.joblib")

model = MLP()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# ---------------------------------------------------------------------------
# Prediccion caso por caso
# ---------------------------------------------------------------------------
for nombre_caso, valores in casos.items():
    entrada = [[valores[col] for col in FEATURES]]
    entrada_escalada = scaler.transform(entrada)
    entrada_t = torch.FloatTensor(entrada_escalada)

    with torch.no_grad():
        probabilidad = model(entrada_t).item()  # ya en probabilidad (Sigmoid aplicada)

    prediccion = "FALLO" if probabilidad >= UMBRAL else "SIN FALLO"

    print(f"--- {nombre_caso} ---")
    print(
        f"  temperatura_cpu={valores['temperatura_cpu']}C, "
        f"uso_cpu={valores['uso_cpu']}%, "
        f"uso_memoria={valores['uso_memoria']}%, "
        f"trafico_red={valores['trafico_red']}Mbps"
    )
    print(f"  Probabilidad de fallo: {probabilidad:.2%}")
    print(f"  Prediccion (umbral {UMBRAL}): {prediccion}")
    print()
