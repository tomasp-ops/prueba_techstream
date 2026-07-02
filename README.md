# TechStream — Detección de anomalías en servidores

Sistema de detección de fallos en servidores basado en dos redes neuronales complementarias, cada una respondiendo a una pregunta distinta:

| | MLP (`mlp_deteccion_tiempo_real/`) | LSTM (`lstm_deteccion_tendencias/`) |
|---|---|---|
| **Pregunta que responde** | ¿Está este servidor en mal estado *ahora mismo*? | ¿Va este servidor camino de fallar, dado cómo ha evolucionado? |
| **Tipo de dato** | Snapshot independiente (un instante) | Serie temporal (40 lecturas consecutivas por servidor) |
| **Frecuencia de ejecución** | Constante, en tiempo real | Periódica (ej. cada hora / cada día) |
| **Coste computacional** | Bajo | Más alto |
| **Resultado en test** | Accuracy 87.9%, Recall 85.4%, Precisión 55.2%, F1 67.0% | Accuracy 100%, Recall 100%, Precisión 100% |

## Por qué dos modelos en vez de uno

Un sistema de detección de anomalías maduro combina ambos enfoques en capas: el MLP actúa como vigilancia rápida y barata sobre el estado presente de cada servidor, mientras que la LSTM analiza tendencias históricas para anticipar fallos que se gestan progresivamente (ej. un memory leak o sobrecalentamiento gradual) antes de que el MLP los detecte como estado crítico. Ninguno sustituye al otro: resuelven problemas distintos.

## Estructura del proyecto

```
prueba_tech/
├── requirements.txt
├── mlp_deteccion_tiempo_real/
│   ├── generate_data.py          # genera datos_servidores.csv (5000 snapshots)
│   ├── eda.py                    # correlaciones, distribuciones, balance de clases
│   ├── model.py                  # clase MLP (nn.Module)
│   ├── train.py                  # entrenamiento (BCEWithLogitsLoss + pos_weight)
│   ├── evaluate.py               # métricas + matriz de confusión
│   ├── model.pt / scaler.joblib  # modelo entrenado final
│   └── imagenes/                 # heatmap, distribuciones, balance, matriz de confusión
│
└── lstm_deteccion_tendencias/
    ├── generate_data_temporal.py # genera datos_servidores_temporal.csv (120 servidores × 40 pasos)
    ├── eda_temporal.py           # balance de clases + trayectorias promedio
    ├── model_lstm.py             # clase LSTMDetector (nn.Module)
    ├── train_lstm.py             # entrenamiento (split por servidor completo)
    ├── evaluate_lstm.py          # métricas + matriz de confusión
    ├── model_lstm.pt / scaler_lstm.joblib
    └── imagenes/                 # balance de clases, trayectorias promedio, matriz de confusión
```

## Cómo ejecutar

```bash
pip install -r requirements.txt

# MLP
cd mlp_deteccion_tiempo_real
python generate_data.py
python eda.py
python train.py
python evaluate.py

# LSTM
cd ../lstm_deteccion_tendencias
python generate_data_temporal.py
python eda_temporal.py
python train_lstm.py
python evaluate_lstm.py
```

## Decisiones de diseño clave

- **Datasets sintéticos con lógica de dominio**: en el MLP, el fallo depende de la combinación de varias métricas en zona de riesgo, no de una sola variable aislada. En la LSTM, el 20% de los servidores sigue una trayectoria de degradación progresiva (subida acelerada de CPU/temperatura) mientras el resto se mantiene estable, partiendo de niveles similares.
- **Arquitecturas construidas explícitamente en PyTorch**, sin usar `sklearn.neural_network` ni wrappers de alto nivel que oculten la gestión de tensores.
- **`pos_weight` en la función de pérdida** para compensar el desbalance de clases (~14-20% de casos positivos en ambos datasets), priorizando el recall sobre la precisión — en detección de fallos, un fallo no detectado es más costoso que una falsa alarma.
- **Split en tres partes (train/validation/test) en el MLP**: la comparación entre distintos valores de `pos_weight` se hizo en el conjunto de validation, no en test, para evitar usar el test set como criterio de decisión (lo cual habría invalidado su función como medida honesta de generalización).
- **`pos_weight` configurable según tolerancia al riesgo**: valor por defecto `3.5` en `mlp_deteccion_tiempo_real/train.py`, elegido por mejor equilibrio precisión/recall (F1 67%, recall 85.42%, precisión 55.16%). Si la prioridad de negocio fuera minimizar al máximo los fallos no detectados, aunque cueste más falsas alarmas, cambiar a `pos_weight=5.98` (proporción negativos/positivos calculada sobre el train set de 5000 filas) sube el recall a 90.28% a costa de bajar la precisión a 44.83%.
- **Split correcto según el tipo de dato**: aleatorio y estratificado por fila en el MLP (snapshots independientes); estratificado por servidor completo en la LSTM (para no filtrar información temporal del mismo servidor entre train y test).
- **El 100% de la LSTM en test se interpreta con cautela**: refleja que el patrón sintético de degradación es muy separable, no necesariamente robustez frente a datos reales más ruidosos.

## Autor

Tomás Pérez — prueba técnica para vacante Técnico Informático IA.
