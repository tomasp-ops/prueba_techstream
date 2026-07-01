"""
generate_data.py
-----------------
Genera un dataset sintetico de telemetria de servidores para practicar
deteccion de fallos. Pensado para explicar en un video paso a paso la
logica de generacion de cada columna.

Columnas generadas:
    - temperatura_cpu (C): 30-95, correlacionada con uso_cpu
    - uso_cpu (%): 5-100
    - uso_memoria (%): 10-100, parcialmente correlacionada con uso_cpu
    - trafico_red (Mbps): 0-1000, mayormente independiente, con picos ocasionales
    - fallo (0/1): raro (~10-15%), solo cuando varias metricas estan
      simultaneamente en zona de riesgo (no por una sola variable aislada)
"""

import numpy as np
import pandas as pd

# Semilla fija para que el dataset sea reproducible entre ejecuciones
np.random.seed(42)

N = 5000  # numero de filas (servidores/observaciones) a generar


def sigmoid(x):
    """Funcion sigmoide clasica: convierte cualquier valor real en una
    probabilidad (0,1). La usamos para transformar un "score de riesgo"
    combinado en una probabilidad de fallo."""
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1) USO_CPU
# ---------------------------------------------------------------------------
# Usamos una distribucion beta(2,2), que es simetrica y acampanada dentro de
# [0,1] (parecida a una normal pero acotada de forma natural, sin necesidad
# de recortar colas). Luego la escalamos al rango realista 5-100 %.
uso_cpu_raw = np.random.beta(a=2, b=2, size=N)
uso_cpu = 5 + uso_cpu_raw * 95  # escala de [0,1] a [5,100]

# ---------------------------------------------------------------------------
# 2) TEMPERATURA_CPU
# ---------------------------------------------------------------------------
# La temperatura depende del uso de CPU: a mas carga, mas calor generado.
# Modelamos una "temperatura base" como funcion lineal de uso_cpu (de ~32C en
# reposo a ~78C a maxima carga) y le sumamos ruido gaussiano para simular
# variabilidad de refrigeracion, ambiente, etc.
temp_base = 32 + (uso_cpu / 100) * 46  # base entre ~32C y ~78C segun la carga
temp_noise = np.random.normal(loc=0, scale=5, size=N)  # ruido termico
temperatura_cpu = temp_base + temp_noise
temperatura_cpu = np.clip(temperatura_cpu, 30, 95)  # forzamos el rango realista

# ---------------------------------------------------------------------------
# 3) USO_MEMORIA
# ---------------------------------------------------------------------------
# La memoria esta parcialmente correlacionada con el uso de CPU (procesos que
# consumen CPU suelen consumir tambien memoria), pero tiene una componente
# propia e independiente (servicios con fugas de memoria, cachés, etc.).
CORR_MEM_CPU = 0.45  # peso de la correlacion con uso_cpu (0=independiente, 1=igual a cpu)
mem_independiente = np.random.normal(loc=55, scale=20, size=N)  # componente propia
mem_noise = np.random.normal(loc=0, scale=6, size=N)  # ruido adicional
uso_memoria = CORR_MEM_CPU * uso_cpu + (1 - CORR_MEM_CPU) * mem_independiente + mem_noise
uso_memoria = np.clip(uso_memoria, 10, 100)

# ---------------------------------------------------------------------------
# 4) TRAFICO_RED
# ---------------------------------------------------------------------------
# El trafico de red es mayormente independiente de CPU/memoria (depende de
# la carga de usuarios/servicios, no del estado interno del servidor).
# Generamos un nivel "normal" con una normal truncada en 0, y anadimos picos
# ocasionales (ráfagas de trafico) en un pequeno porcentaje de filas.
trafico_base = np.abs(np.random.normal(loc=200, scale=150, size=N))

PROB_PICO = 0.06  # ~6% de las observaciones tienen un pico de trafico
es_pico = np.random.rand(N) < PROB_PICO
magnitud_pico = np.random.uniform(low=500, high=900, size=N)

trafico_red = trafico_base + np.where(es_pico, magnitud_pico, 0)
trafico_red = np.clip(trafico_red, 0, 1000)

# ---------------------------------------------------------------------------
# 5) FALLO (variable objetivo)
# ---------------------------------------------------------------------------
# Queremos que el fallo sea raro y que dependa de la COMBINACION de varias
# metricas en zona de riesgo simultaneamente, no de una sola variable alta.
#
# Estrategia:
#   a) Definimos "zonas de riesgo" (flags binarios) para cada metrica.
#   b) Contamos cuantas zonas de riesgo estan activas a la vez (0 a 4).
#   c) Ese conteo es el termino dominante del logit: cada zona de riesgo
#      adicional dispara la probabilidad de forma no lineal (via sigmoide),
#      de modo que 1 sola metrica en riesgo apenas mueve la probabilidad,
#      pero 2-3 simultaneas la disparan.
#   d) Anadimos tambien una contribucion continua (mas suave) de cada
#      variable para que no sea un escalon puro, sino una transicion realista.
riesgo_temp = temperatura_cpu > 70
riesgo_cpu = uso_cpu > 75
riesgo_mem = uso_memoria > 75
riesgo_trafico = trafico_red > 550

conteo_riesgos = (
    riesgo_temp.astype(int)
    + riesgo_cpu.astype(int)
    + riesgo_mem.astype(int)
    + riesgo_trafico.astype(int)
)

# Logit = combinacion ponderada. El sesgo (-5) hace que con 0 riesgos activos
# la probabilidad sea casi nula (~1%); con 1 riesgo suba un poco (~15-20%,
# sigue siendo minoritario); con 2 riesgos se dispare (~70%); con 3-4 sea
# casi seguro (~95-100%). Los terminos continuos (pequenos) anaden
# variabilidad fina dentro de cada nivel de riesgo.
logit = (
    -5.0
    + 2.5 * conteo_riesgos
    + 0.015 * (temperatura_cpu - 60)
    + 0.015 * (uso_cpu - 50)
    + 0.008 * (uso_memoria - 50)
    + 0.003 * (trafico_red - 200)
)

prob_fallo = sigmoid(logit)
fallo = np.random.binomial(n=1, p=prob_fallo)

# ---------------------------------------------------------------------------
# Ensamblado del DataFrame final y guardado a CSV
# ---------------------------------------------------------------------------
df = pd.DataFrame(
    {
        "temperatura_cpu": temperatura_cpu.round(2),
        "uso_cpu": uso_cpu.round(2),
        "uso_memoria": uso_memoria.round(2),
        "trafico_red": trafico_red.round(2),
        "fallo": fallo,
    }
)

df.to_csv("datos_servidores.csv", index=False)

print(f"CSV generado: datos_servidores.csv ({len(df)} filas)")
print(f"Tasa de fallo: {df['fallo'].mean():.2%}")
print()
print("Primeras filas:")
print(df.head())
print()
print("Describe:")
print(df.describe())
