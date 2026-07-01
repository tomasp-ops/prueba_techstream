"""
eda.py
------
Analisis exploratorio (EDA) del dataset sintetico de telemetria de
servidores. Genera 3 graficos para apoyar la explicacion en video:

    1) correlacion_heatmap.png - correlacion de Pearson entre las 5 columnas
    2) distribuciones.png      - histograma de cada variable numerica,
                                 coloreado por la clase fallo (0/1)
    3) balance_clases.png      - conteo de fallo (0 vs 1), para ver el
                                 desbalance de clases
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, solo para guardar archivos PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMAGENES_DIR = "imagenes"
os.makedirs(IMAGENES_DIR, exist_ok=True)

# Intentamos usar seaborn para el heatmap (mas bonito), pero si no esta
# instalado, hacemos un fallback con matplotlib puro (imshow + texto manual).
try:
    import seaborn as sns

    TIENE_SEABORN = True
except ImportError:
    TIENE_SEABORN = False

df = pd.read_csv("datos_servidores.csv")

COLUMNAS_NUMERICAS = ["temperatura_cpu", "uso_cpu", "uso_memoria", "trafico_red"]

# ---------------------------------------------------------------------------
# 1) HEATMAP DE CORRELACION DE PEARSON (5 columnas, incluyendo fallo)
# ---------------------------------------------------------------------------
corr = df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(7, 6))
if TIENE_SEABORN:
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
else:
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    # Anotamos manualmente el valor numerico en cada celda
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(im, ax=ax)

ax.set_title("Correlacion de Pearson entre variables")
fig.tight_layout()
fig.savefig(os.path.join(IMAGENES_DIR, "correlacion_heatmap.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) HISTOGRAMAS 2x2, coloreados por fallo (0 vs 1)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

df_sin_fallo = df[df["fallo"] == 0]
df_con_fallo = df[df["fallo"] == 1]

for ax, col in zip(axes, COLUMNAS_NUMERICAS):
    bins = np.histogram_bin_edges(df[col], bins=30)
    ax.hist(df_sin_fallo[col], bins=bins, alpha=0.6, label="fallo=0", color="steelblue")
    ax.hist(df_con_fallo[col], bins=bins, alpha=0.6, label="fallo=1", color="crimson")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("frecuencia")
    ax.legend()

fig.suptitle("Distribucion de cada variable, separada por fallo")
fig.tight_layout()
fig.savefig(os.path.join(IMAGENES_DIR, "distribuciones.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) GRAFICO DE BARRAS: conteo de fallo (0 vs 1) -> desbalance de clases
# ---------------------------------------------------------------------------
conteo_fallo = df["fallo"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 5))
barras = ax.bar(
    conteo_fallo.index.astype(str),
    conteo_fallo.values,
    color=["steelblue", "crimson"],
)
ax.set_xlabel("fallo")
ax.set_ylabel("numero de filas")
ax.set_title("Balance de clases: fallo (0 vs 1)")
for barra, valor in zip(barras, conteo_fallo.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        str(valor),
        ha="center",
        va="bottom",
    )
fig.tight_layout()
fig.savefig(os.path.join(IMAGENES_DIR, "balance_clases.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# RESUMEN INTERPRETATIVO (para leer en el video)
# ---------------------------------------------------------------------------
corr_fallo = corr["fallo"].drop("fallo").sort_values(ascending=False)
pct_fallo = df["fallo"].mean() * 100

print("Resumen del EDA:")
print(
    f"- El dataset esta desbalanceado: solo el {pct_fallo:.1f}% de las filas son fallo=1, "
    "por lo que un modelo debe evaluarse con metricas como recall/F1, no solo accuracy."
)
print(
    f"- La variable mas correlacionada con el fallo es '{corr_fallo.index[0]}' "
    f"(r={corr_fallo.iloc[0]:.2f}), seguida de '{corr_fallo.index[1]}' (r={corr_fallo.iloc[1]:.2f}); "
    "ninguna correlacion individual es muy alta, lo que confirma que el fallo depende de la "
    "combinacion de varias metricas y no de una sola variable aislada."
)
print(
    "- En los histogramas, la distribucion de fallo=1 (rojo) esta desplazada hacia valores altos "
    "de temperatura, uso de CPU y memoria respecto a fallo=0 (azul), mientras que en trafico_red "
    "la separacion es mas leve, coherente con su menor correlacion."
)
print(
    "- El grafico de barras confirma visualmente el desbalance de clases, algo a tener en cuenta "
    "al entrenar y validar cualquier modelo de clasificacion sobre este dataset."
)
