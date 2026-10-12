"""Interfaz interactiva - Detección de anomalías de vibración (RED-001).
Ejecutar:  streamlit run app_streamlit.py
Requiere: datos.csv (o MBC_Datos_Sinteticos_Prueba.csv) en la misma carpeta.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor

st.set_page_config(page_title="Anomalías de vibración", layout="wide")
CENTROS = np.array([420, 600, 850, 1110, 1340])


@st.cache_data
def cargar(ruta):
    df = pd.read_csv(ruta, parse_dates=["fecha_hora"])
    return df.rename(columns={"vibracion_rms_mm_s": "vib", "temperatura_c": "temp"})


def analizar(df, umbral):
    c = df.dropna(subset=["rpm", "vib"]).copy()
    hub = HuberRegressor().fit(c[["rpm"]], c.vib)
    c["esp"] = hub.predict(c[["rpm"]])
    c["res"] = c.vib - c.esp
    med = c.res.median()
    sigma = 1.4826 * (c.res - med).abs().median()
    c["z"] = (c.res - med) / sigma
    c["anom"] = c.z.abs() > umbral
    c["reg"] = [f"~{CENTROS[np.abs(CENTROS - r).argmin()]} rpm" for r in c.rpm]
    return c, hub, sigma, med


# ---------------- Barra lateral ----------------
st.sidebar.header("Controles")
ruta = "datos.csv" if os.path.exists("datos.csv") else "MBC_Datos_Sinteticos_Prueba.csv"
datos = cargar(ruta)
umbral = st.sidebar.slider("Umbral de |z| robusto", 2.0, 5.0, 3.5, 0.1)
f_ini, f_fin = st.sidebar.date_input(
    "Rango de fechas", (datos.fecha_hora.min().date(), datos.fecha_hora.max().date()))
regs = st.sidebar.multiselect("Régimen de operación", [f"~{c} rpm" for c in CENTROS],
                              default=[f"~{c} rpm" for c in CENTROS])
solo_anom = st.sidebar.checkbox("Mostrar solo anomalías en la tabla", value=False)

c, hub, sigma, med = analizar(datos, umbral)
m = (c.fecha_hora.dt.date >= f_ini) & (c.fecha_hora.dt.date <= f_fin) & c.reg.isin(regs)
v = c[m]

# ---------------- Encabezado y métricas ----------------
st.title("Identificación de anomalías de vibración — RED-001")
st.caption("Vibración esperada condicionada a las RPM (regresión robusta de Huber) "
           "y z-score robusto (mediana y MAD) sobre los residuos.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros evaluados", len(v))
k2.metric("Anomalías", f"{int(v.anom.sum())}  ({v.anom.mean():.1%})" if len(v) else "0")
k3.metric("σ robusta (mm/s)", f"{sigma:.3f}")
k4.metric("Pendiente (mm/s por 1000 rpm)", f"+{hub.coef_[0]*1000:.2f}")

# ---------------- Gráfico 3 interactivo ----------------
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Vibración vs RPM (rojo = anómalo)", "z robusto en el tiempo"), horizontal_spacing=.08)
n, a = v[~v.anom], v[v.anom]
hover = "%{customdata[0]}<br>RPM: %{x}<br>Vib: %{y:.3f} mm/s<br>z: %{customdata[1]:.1f}<extra></extra>"
fig.add_trace(go.Scatter(x=n.rpm, y=n.vib, mode="markers", name="Normal",
                         marker=dict(color="#1f77b4", size=6, opacity=.75),
                         customdata=np.c_[n.fecha_hora.dt.strftime("%Y-%m-%d %H:%M"), n.z],
                         hovertemplate=hover), 1, 1)
fig.add_trace(go.Scatter(x=a.rpm, y=a.vib, mode="markers", name="Anómalo",
                         marker=dict(color="#d62728", size=10, symbol="x"),
                         customdata=np.c_[a.fecha_hora.dt.strftime("%Y-%m-%d %H:%M"), a.z],
                         hovertemplate=hover), 1, 1)
r = np.linspace(c.rpm.min(), c.rpm.max(), 50)
esp = hub.intercept_ + hub.coef_[0] * r
fig.add_trace(go.Scatter(x=r, y=esp, mode="lines", name="Esperada (Huber)",
                         line=dict(color="green")), 1, 1)
for s in (umbral, -umbral):
    fig.add_trace(go.Scatter(x=r, y=esp + med + s * sigma, mode="lines", showlegend=False,
                             line=dict(color="red", dash="dash", width=1)), 1, 1)
fig.add_trace(go.Scatter(x=n.fecha_hora, y=n.z, mode="markers", showlegend=False,
                         marker=dict(color="#1f77b4", size=5, opacity=.75)), 1, 2)
fig.add_trace(go.Scatter(x=a.fecha_hora, y=a.z, mode="markers", showlegend=False,
                         marker=dict(color="#d62728", size=8, symbol="x")), 1, 2)
for s in (umbral, -umbral):
    fig.add_hline(y=s, line=dict(color="red", dash="dash", width=1), row=1, col=2)
fig.update_xaxes(title_text="RPM", row=1, col=1)
fig.update_yaxes(title_text="Vibración RMS (mm/s)", row=1, col=1)
fig.update_xaxes(title_text="Fecha", row=1, col=2)
fig.update_yaxes(title_text="z robusto", row=1, col=2)
fig.update_layout(height=470, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-.18))
st.plotly_chart(fig, width="stretch")

# ---------------- Pestañas ----------------
t1, t2, t3 = st.tabs(["Vibración en el tiempo", "Distribución por régimen", "Tabla de registros"])
with t1:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=v.fecha_hora, y=v.vib, mode="lines", name="Vibración",
                            line=dict(color="gray", width=1)))
    f2.add_trace(go.Scatter(x=v.fecha_hora, y=v.esp, mode="lines", name="Esperada",
                            line=dict(color="green", width=1.5, dash="dot")))
    f2.add_trace(go.Scatter(x=v[v.anom].fecha_hora, y=v[v.anom].vib, mode="markers",
                            name="Anómalo", marker=dict(color="#d62728", size=8, symbol="x")))
    f2.update_layout(height=380, xaxis_title="Fecha", yaxis_title="Vibración RMS (mm/s)",
                     margin=dict(t=20))
    st.plotly_chart(f2, width="stretch")
with t2:
    f3 = go.Figure()
    for reg in sorted(v.reg.unique(), key=lambda s: int(s[1:-4])):
        g = v[(v.reg == reg) & ~v.anom]
        f3.add_trace(go.Box(y=g.vib, name=reg, boxpoints="outliers"))
    ga = v[v.anom]
    f3.add_trace(go.Scatter(x=ga.reg, y=ga.vib, mode="markers", name="Anómalos",
                            marker=dict(color="#d62728", size=8, symbol="x")))
    f3.update_layout(height=380, yaxis_title="Vibración RMS (mm/s)", margin=dict(t=20))
    st.plotly_chart(f3, width="stretch")
    st.caption("Las cajas resumen solo los registros normales; los anómalos se superponen en rojo.")
with t3:
    tab = (v[v.anom] if solo_anom else v)[["fecha_hora", "rpm", "vib", "esp", "res", "z", "temp", "anom"]]
    st.dataframe(tab.rename(columns={"vib": "vibración", "esp": "esperada", "res": "residuo"}),
                 width="stretch", height=360)
    st.download_button("Descargar CSV", tab.to_csv(index=False).encode(), "registros.csv", "text/csv")
