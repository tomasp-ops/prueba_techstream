"""
model.py
--------
Define la arquitectura de red neuronal (MLP) usada para clasificar si un
servidor va a fallar (fallo=0/1) a partir de sus 4 metricas de telemetria:
temperatura_cpu, uso_cpu, uso_memoria, trafico_red.

Este archivo NO entrena el modelo, solo define la arquitectura y hace una
prueba de forward pass con datos aleatorios para verificar que las
dimensiones de los tensores fluyen correctamente por la red.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Red neuronal totalmente conectada (Multi-Layer Perceptron) para
    clasificacion binaria de fallos de servidor.

    Arquitectura y por que se eligio asi:
        - Entrada (4 -> 16): tenemos solo 4 features numericas (temperatura,
          uso de CPU, uso de memoria, trafico de red). Expandir a 16
          neuronas le da a la red suficiente capacidad para aprender
          combinaciones no lineales entre esas 4 variables (recordemos que
          el fallo depende de la INTERACCION entre varias metricas a la vez,
          no de una sola), sin ser excesivo para un problema tan pequeno.
        - Oculta (16 -> 8): una segunda capa que reduce la dimensionalidad
          progresivamente ("embudo"), forzando a la red a quedarse con una
          representacion mas compacta y abstracta de esas combinaciones
          antes de decidir. Con un dataset de ~2000 filas y solo 4 features,
          dos capas ocultas pequenas (16 y 8) son suficientes: mas capas o
          mas neuronas aumentarian el riesgo de overfitting sin aportar
          capacidad util.
        - ReLU en las capas ocultas: es la activacion estandar por defecto
          para capas intermedias porque es simple, barata de calcular y evita
          el problema de gradientes que se desvanecen (a diferencia de
          sigmoid/tanh en redes profundas). Introduce la no linealidad
          necesaria para que la red pueda aprender fronteras de decision
          complejas (no solo combinaciones lineales de las 4 features).
        - Salida (8 -> 1) con Sigmoid: como es clasificacion BINARIA (fallo
          si/no), necesitamos una unica neurona de salida cuyo valor podamos
          interpretar directamente como una probabilidad entre 0 y 1. Sigmoid
          comprime cualquier valor real a ese rango (0,1), a diferencia de
          softmax (pensado para multi-clase) o de no usar activacion
          (que daria un logit sin interpretacion probabilistica directa).
    """

    def __init__(self):
        super().__init__()
        self.capa_entrada = nn.Linear(in_features=4, out_features=16)
        self.capa_oculta = nn.Linear(in_features=16, out_features=8)
        self.capa_salida = nn.Linear(in_features=8, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        # x tiene forma (batch_size, 4): un lote de servidores, cada uno
        # con sus 4 metricas de telemetria.

        # Capa 1: 4 -> 16, seguida de ReLU para introducir no linealidad.
        x = self.capa_entrada(x)
        x = self.relu(x)

        # Capa 2 (oculta): 16 -> 8, tambien con ReLU.
        x = self.capa_oculta(x)
        x = self.relu(x)

        # Capa de salida: 8 -> 1. Esto son "logits" (valores reales sin
        # acotar), no probabilidades todavia.
        logits = self.capa_salida(x)

        if return_logits:
            # nn.BCEWithLogitsLoss espera logits crudos (aplica Sigmoid
            # internamente de forma numericamente mas estable que hacerlo a
            # mano). Se usa asi durante el entrenamiento.
            return logits

        # Para inferencia/evaluacion normal, devolvemos la probabilidad ya
        # pasada por Sigmoid, que es lo que se espera al usar el modelo.
        return self.sigmoid(logits)


if __name__ == "__main__":
    # Prueba rapida: no entrenamos nada todavia, solo verificamos que las
    # dimensiones fluyen bien por la red con pesos inicializados al azar.
    torch.manual_seed(42)

    modelo = MLP()
    print(modelo)
    print()

    # Batch de 5 ejemplos "falsos" con 4 features cada uno.
    batch_falso = torch.randn(5, 4)
    print("Input (5, 4):")
    print(batch_falso)
    print()

    salida = modelo(batch_falso)
    print("Output (5, 1) - probabilidades de fallo:")
    print(salida)
    print()
    print(f"Forma de salida: {tuple(salida.shape)}")
