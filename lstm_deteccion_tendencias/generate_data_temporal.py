"""
generate_data_temporal.py
--------------------------
Genera un dataset sintetico de telemetria de servidores en formato de SERIE
TEMPORAL (pensado para un modelo LSTM), a diferencia del dataset de
snapshots sueltos usado para el MLP.

Cada servidor se observa durante 40 pasos temporales consecutivos ("horas").
Se simulan dos tipos de servidor:

    - Servidores ESTABLES (80%, 96 de 120): fluctuan alrededor de un valor
      base propio, sin ninguna tendencia, y nunca fallan.
    - Servidores en DEGRADACION (20%, 24 de 120): parten de un estado
      normal pero sus metricas (sobre todo CPU y temperatura) suben de
      forma progresiva paso a paso hasta terminar en zona de fallo en los
      ultimos pasos de la secuencia.

Formato de salida: "long format" (una fila por servidor por paso temporal),
con columnas: id_servidor, paso_temporal, temperatura_cpu, uso_cpu,
uso_memoria, trafico_red, fallo.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducibilidad

N_SERVIDORES = 120
N_PASOS = 40  # 40 lecturas consecutivas ("horas") por servidor
PCT_DEGRADACION = 0.20
N_DEGRADACION = round(N_SERVIDORES * PCT_DEGRADACION)  # 24 servidores

# Elegimos al azar que 24 servidores (de los 120) seran los que degradan.
# Usamos una permutacion en vez de simplemente los primeros 24 para que el
# tipo de servidor no dependa de su id (mas realista y evita cualquier sesgo
# si mas adelante se usa el id_servidor como feature).
ids_degradacion = set(np.random.permutation(N_SERVIDORES)[:N_DEGRADACION].tolist())

filas = []

for id_servidor in range(N_SERVIDORES):
    es_degradacion = id_servidor in ids_degradacion
    pasos = np.arange(N_PASOS)

    if es_degradacion:
        # -------------------------------------------------------------
        # SERVIDOR EN DEGRADACION
        # -------------------------------------------------------------
        # Idea: el servidor arranca en un estado de carga normal (como
        # cualquier servidor estable) pero tiene una "deriva" (drift)
        # positiva que lo empuja, paso a paso, hacia valores cada vez mas
        # altos de CPU y temperatura, hasta terminar en zona de riesgo.
        #
        # En vez de una recta perfecta (que seria irrealista), construimos
        # la tendencia con:
        #   1) una curva de progreso no lineal (progreso**1.4): al elevar
        #      el progreso [0,1] a una potencia > 1, el crecimiento es mas
        #      lento al principio y se acelera hacia el final, imitando
        #      como un problema real (fuga termica, proceso descontrolado,
        #      etc.) suele agravarse cada vez mas rapido cerca del fallo.
        #   2) ruido gaussiano por paso, para que la subida no sea una
        #      linea recta sino que tenga altibajos realistas.
        cpu_base = np.random.uniform(25, 45)  # arranca en carga normal
        cpu_incremento_total = np.random.uniform(45, 65)  # cuanto sube en total
        progreso = (pasos / (N_PASOS - 1)) ** 1.4  # 0 -> 1, acelerando al final
        cpu_tendencia = cpu_base + progreso * cpu_incremento_total
        cpu_ruido = np.random.normal(0, 4, N_PASOS)
        uso_cpu = np.clip(cpu_tendencia + cpu_ruido, 5, 100)

        # La memoria sube de forma mas leve: solo esta parcialmente
        # correlacionada con la CPU (igual que en el dataset de snapshots),
        # asi que "arrastra" algo de la subida de CPU pero tiene su propia
        # componente de ruido independiente.
        mem_base_srv = np.random.uniform(35, 55)
        mem_ruido = np.random.normal(0, 6, N_PASOS)
        uso_memoria = np.clip(0.45 * uso_cpu + 0.55 * (mem_base_srv + mem_ruido), 10, 100)

    else:
        # -------------------------------------------------------------
        # SERVIDOR ESTABLE
        # -------------------------------------------------------------
        # Sin tendencia: fluctua con ruido gaussiano alrededor de un valor
        # base fijo propio de ese servidor, durante los 40 pasos.
        cpu_base_srv = np.random.uniform(20, 55)
        cpu_ruido = np.random.normal(0, 4, N_PASOS)
        uso_cpu = np.clip(cpu_base_srv + cpu_ruido, 5, 100)

        mem_base_srv = np.random.uniform(35, 65)
        mem_ruido = np.random.normal(0, 6, N_PASOS)
        uso_memoria = np.clip(0.45 * uso_cpu + 0.55 * (mem_base_srv + mem_ruido), 10, 100)

    # La temperatura depende de la CPU en ambos casos (misma logica que en
    # el dataset de snapshots): a mas CPU, mas temperatura base, mas ruido.
    # Como para los servidores en degradacion uso_cpu ya sube con el tiempo,
    # la temperatura sube "en cascada" de forma natural, sin necesidad de
    # una tendencia propia adicional.
    temp_base = 32 + (uso_cpu / 100) * 46
    temp_ruido = np.random.normal(0, 4, N_PASOS)
    temperatura_cpu = np.clip(temp_base + temp_ruido, 30, 95)

    # El trafico de red es mayormente independiente del estado interno del
    # servidor (igual en ambos tipos), con picos ocasionales de rafaga.
    trafico_base = np.abs(np.random.normal(180, 120, N_PASOS))
    es_pico = np.random.rand(N_PASOS) < 0.05
    magnitud_pico = np.random.uniform(400, 800, N_PASOS)
    trafico_red = np.clip(trafico_base + np.where(es_pico, magnitud_pico, 0), 0, 1000)

    # El fallo se marca UNICAMENTE en el ultimo paso de la secuencia
    # (paso_temporal = 39), y solo para los servidores que "terminan
    # fallando" (los de degradacion). Los servidores estables nunca fallan.
    fallo = np.zeros(N_PASOS, dtype=int)
    if es_degradacion:
        fallo[-1] = 1

    for paso in range(N_PASOS):
        filas.append(
            {
                "id_servidor": id_servidor,
                "paso_temporal": paso,
                "temperatura_cpu": round(temperatura_cpu[paso], 2),
                "uso_cpu": round(uso_cpu[paso], 2),
                "uso_memoria": round(uso_memoria[paso], 2),
                "trafico_red": round(trafico_red[paso], 2),
                "fallo": int(fallo[paso]),
            }
        )

df = pd.DataFrame(filas)
df.to_csv("datos_servidores_temporal.csv", index=False)

# ---------------------------------------------------------------------------
# Resumen para verificar la generacion
# ---------------------------------------------------------------------------
print(f"CSV generado: datos_servidores_temporal.csv ({len(df)} filas)")
print(f"Servidores: {N_SERVIDORES} ({N_DEGRADACION} en degradacion, {N_SERVIDORES - N_DEGRADACION} estables)")
print()

print("Primeras 10 filas:")
print(df.head(10))
print()

fallo_por_servidor = df.groupby("id_servidor")["fallo"].max()
n_terminan_fallo = (fallo_por_servidor == 1).sum()
n_no_fallo = (fallo_por_servidor == 0).sum()
print(f"Servidores que terminan en fallo: {n_terminan_fallo}")
print(f"Servidores que NO fallan: {n_no_fallo}")
print()

# Ejemplo: evolucion de uso_cpu para un servidor en degradacion y uno estable
id_ejemplo_degradacion = sorted(ids_degradacion)[0]
id_ejemplo_estable = min(set(range(N_SERVIDORES)) - ids_degradacion)

cpu_degradacion = df[df["id_servidor"] == id_ejemplo_degradacion]["uso_cpu"].tolist()
cpu_estable = df[df["id_servidor"] == id_ejemplo_estable]["uso_cpu"].tolist()

print(f"Evolucion de uso_cpu - servidor {id_ejemplo_degradacion} (DEGRADACION, termina en fallo):")
print(cpu_degradacion)
print()
print(f"Evolucion de uso_cpu - servidor {id_ejemplo_estable} (ESTABLE, sin fallo):")
print(cpu_estable)
