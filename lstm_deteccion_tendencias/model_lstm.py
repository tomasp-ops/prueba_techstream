"""
model_lstm.py
-------------
Define la arquitectura LSTM usada para predecir si un servidor terminara
fallando a partir de su HISTORIAL de 40 lecturas consecutivas (secuencia
temporal), a diferencia del MLP que clasifica un snapshot suelto sin
memoria del pasado.

Este archivo NO entrena el modelo, solo define la arquitectura (ver
train_lstm.py para el entrenamiento).
"""

import torch
import torch.nn as nn


class LSTMDetector(nn.Module):
    """Red LSTM para clasificacion binaria sobre secuencias temporales de
    telemetria de servidor.

    Por que LSTM y no un MLP aqui:
        El dataset temporal no son snapshots sueltos, sino secuencias de 40
        pasos por servidor donde lo relevante es la TENDENCIA a lo largo del
        tiempo (una CPU que sube progresivamente durante 40 horas es una
        senal de alarma distinta a una CPU alta en un unico instante). Un
        MLP procesaria cada paso de forma independiente y no podria
        "recordar" la evolucion previa. Una LSTM, en cambio, procesa la
        secuencia paso a paso y mantiene un estado interno (hidden state)
        que se va actualizando con cada nueva lectura, acumulando
        informacion sobre la trayectoria completa del servidor.

    Que representa hidden_size (32):
        Es la dimensionalidad del "resumen" interno que la LSTM mantiene y
        actualiza en cada paso temporal (el hidden state). Cuanto mayor es
        hidden_size, mas patrones distintos de evolucion temporal puede
        llegar a representar la red, a costa de mas parametros a entrenar.
        32 es un tamano pequeno pero razonable para un problema con solo 4
        features de entrada y ~100 secuencias de entrenamiento: suficiente
        capacidad para capturar la tendencia de subida/estabilidad, sin
        sobreajustar con tan pocos datos.

    Estructura:
        - nn.LSTM(input_size=4, hidden_size=32, num_layers=1,
          batch_first=True): recorre la secuencia de 40 pasos, cada uno con
          4 features, y produce un hidden state de tamano 32 en cada paso.
        - Solo nos quedamos con el ULTIMO hidden state de la secuencia (el
          que ha visto ya las 40 lecturas completas), porque es el que
          resume "todo lo que le paso a este servidor" y es sobre el que
          decidimos si termino fallando o no.
        - nn.Linear(32, 1) + Sigmoid: igual que en el MLP, comprime ese
          resumen de 32 numeros a una unica probabilidad de fallo (0-1).
    """

    def __init__(self, input_size=4, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.capa_salida = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        # x tiene forma (batch_size, 40, 4): un lote de servidores, cada uno
        # con su secuencia de 40 pasos temporales y 4 features por paso.

        # salida: hidden state de CADA paso temporal, forma (batch, 40, 32)
        # h_n: ULTIMO hidden state de cada capa, forma (num_layers, batch, 32)
        salida, (h_n, c_n) = self.lstm(x)

        # Nos quedamos solo con el ultimo hidden state de la ultima capa,
        # que resume toda la secuencia ya procesada. Con num_layers=1,
        # h_n[-1] equivale al hidden state del ultimo paso temporal
        # (salida[:, -1, :]), pero usamos h_n para dejar explicito que es
        # el "estado final" de la LSTM el que alimenta la clasificacion.
        ultimo_hidden = h_n[-1]  # forma (batch, 32)

        logits = self.capa_salida(ultimo_hidden)  # forma (batch, 1)

        if return_logits:
            # BCEWithLogitsLoss espera logits crudos (ver train_lstm.py)
            return logits

        return self.sigmoid(logits)


if __name__ == "__main__":
    # Prueba rapida de forma, igual que en model.py: un batch falso de 5
    # servidores con 40 pasos y 4 features cada uno.
    torch.manual_seed(42)

    modelo = LSTMDetector()
    print(modelo)

    batch_falso = torch.randn(5, 40, 4)
    salida = modelo(batch_falso)
    print(f"Input: {tuple(batch_falso.shape)} -> Output: {tuple(salida.shape)}")
    print(salida)
