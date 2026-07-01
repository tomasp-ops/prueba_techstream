"""
eda_temporal.py
----------------
Analisis exploratorio (EDA) del dataset temporal de telemetria de
servidores. Genera 2 graficos para apoyar la explicacion en video:

    1) balance_clases_lstm.png  - conteo de servidores por resultado final
                                   (fallo vs no fallo)
    2) trayectorias_promedio.png - evolucion promedio de uso_cpu y
                                   temperatura_cpu a lo largo de los 40
                                   pasos, comparando el grupo "degradacion"
                                   vs el grupo "estable"
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IMAGENES_DIR = "imagenes"
os.makedirs(IMAGENES_DIR, exist_ok=True)

df = pd.read_csv("datos_servidores_temporal.csv")

# Etiqueta por servidor: 1 si fallo en algun paso de su secuencia (el grupo
# "degradacion"), 0 si nunca fallo (el grupo "estable"). Se hace con un
# groupby + max porque "fallo" solo esta marcado en el ultimo paso.
resultado_por_servidor = df.groupby("id_servidor")["fallo"].max()

ids_degradacion = resultado_por_servidor[resultado_por_servidor == 1].index
ids_estable = resultado_por_servidor[resultado_por_servidor == 0].index

# ---------------------------------------------------------------------------
# 1) Grafico de barras: conteo de servidores por resultado final
# ---------------------------------------------------------------------------
conteo = resultado_por_servidor.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 5))
barras = ax.bar(
    ["0 (estable)", "1 (fallo)"],
    [conteo.get(0, 0), conteo.get(1, 0)],
    color=["steelblue", "crimson"],
)
ax.set_xlabel("resultado del servidor")
ax.set_ylabel("numero de servidores")
ax.set_title("Balance de clases: servidores con fallo vs sin fallo")
for barra, valor in zip(barras, [conteo.get(0, 0), conteo.get(1, 0)]):
    ax.text(barra.get_x() + barra.get_width() / 2, valor, str(valor), ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(IMAGENES_DIR, "balance_clases_lstm.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) Trayectorias promedio de uso_cpu y temperatura_cpu por paso temporal,
#    separando el grupo degradacion del grupo estable.
# ---------------------------------------------------------------------------
df_degradacion = df[df["id_servidor"].isin(ids_degradacion)]
df_estable = df[df["id_servidor"].isin(ids_estable)]

cpu_prom_degradacion = df_degradacion.groupby("paso_temporal")["uso_cpu"].mean()
cpu_prom_estable = df_estable.groupby("paso_temporal")["uso_cpu"].mean()

temp_prom_degradacion = df_degradacion.groupby("paso_temporal")["temperatura_cpu"].mean()
temp_prom_estable = df_estable.groupby("paso_temporal")["temperatura_cpu"].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(cpu_prom_degradacion.index, cpu_prom_degradacion.values, label="degradacion", color="crimson")
axes[0].plot(cpu_prom_estable.index, cpu_prom_estable.values, label="estable", color="steelblue")
axes[0].set_title("uso_cpu promedio por paso temporal")
axes[0].set_xlabel("paso_temporal")
axes[0].set_ylabel("uso_cpu (%)")
axes[0].legend()

axes[1].plot(temp_prom_degradacion.index, temp_prom_degradacion.values, label="degradacion", color="crimson")
axes[1].plot(temp_prom_estable.index, temp_prom_estable.values, label="estable", color="steelblue")
axes[1].set_title("temperatura_cpu promedio por paso temporal")
axes[1].set_xlabel("paso_temporal")
axes[1].set_ylabel("temperatura_cpu (C)")
axes[1].legend()

fig.suptitle("Trayectorias promedio: degradacion vs estable")
fig.tight_layout()
fig.savefig(os.path.join(IMAGENES_DIR, "trayectorias_promedio.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Resumen interpretativo (para leer en el video)
# ---------------------------------------------------------------------------
cpu_inicio_deg = cpu_prom_degradacion.iloc[0]
cpu_final_deg = cpu_prom_degradacion.iloc[-1]
cpu_inicio_est = cpu_prom_estable.iloc[0]
cpu_final_est = cpu_prom_estable.iloc[-1]

print("Resumen del EDA temporal:")
print(
    f"- El dataset esta desbalanceado por servidor: {len(ids_degradacion)} de {len(resultado_por_servidor)} "
    f"servidores terminan en fallo ({len(ids_degradacion) / len(resultado_por_servidor):.1%}), consistente con "
    "el ~20% de servidores en degradacion definido al generar los datos."
)
print(
    f"- El grupo 'estable' se mantiene plano en torno a un uso_cpu promedio de ~{cpu_inicio_est:.1f}%-{cpu_final_est:.1f}% "
    f"durante los 40 pasos, mientras que el grupo 'degradacion' arranca en un nivel similar (~{cpu_inicio_deg:.1f}%) "
    f"pero termina disparado hasta ~{cpu_final_deg:.1f}% de media, confirmando visualmente la tendencia de subida "
    "progresiva que se penso al generar los datos."
)
print(
    "- La temperatura sigue el mismo patron que la CPU (logico, ya que esta correlacionada con ella): se "
    "mantiene estable en el grupo sin fallo y sube en paralelo a la CPU en el grupo de degradacion, lo que "
    "sugiere que ambas variables combinadas son una senal temporal clara para distinguir los dos grupos."
)
