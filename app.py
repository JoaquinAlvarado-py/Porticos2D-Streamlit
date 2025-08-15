# -*- coding: utf-8 -*-
# Pórticos 2D – Análisis Modal (OpenSeesPy) – m–kg–s
# Autor: tú :)  | Comentarios y UI en español

import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # entorno sin GUI (Streamlit Cloud)
import matplotlib.pyplot as plt

# --- Importaciones tolerantes ---
OPS_OK, OPSVIS_OK = True, True
OPS_IMPORT_ERROR, OPSVIS_IMPORT_ERROR = "", ""

try:
    import openseespy.opensees as ops
except Exception as e:
    OPS_OK, OPS_IMPORT_ERROR = False, str(e)

try:
    import opsvis as opsv
except Exception as e:
    OPSVIS_OK, OPSVIS_IMPORT_ERROR = False, str(e)

# -----------------------------------------------------------------------------
# Núcleo de análisis
# -----------------------------------------------------------------------------
def armar_y_analizar(niveles:int, panos:int,
                     h_nivel:float, luz_viga:float,
                     m_nodal:float,
                     E:float, A_col:float, Iz_col:float, A_viga:float, Iz_viga:float,
                     modos:int|None=None):
    """
    Construye un pórtico 2D regular y hace un análisis modal.
    Unidades: m–kg–s (E en Pa).
    - niveles: cantidad de pisos
    - panos: cantidad de paños
    - h_nivel: altura por nivel [m]
    - luz_viga: separación entre ejes de columnas [m]
    - m_nodal: masa concentrada en nodos de cada nivel [kg]
    - E: módulo de elasticidad [Pa]
    - A_col, Iz_col, A_viga, Iz_viga: propiedades geométricas [m²], [m⁴]
    - modos: nº de modos a extraer (por defecto = niveles)
    Retorna: (T, rutas_imagenes, msg_error)
    """
    if not OPS_OK:
        return None, None, ("OpenSeesPy no está disponible.\n" +
                            f"Detalle de importación: {OPS_IMPORT_ERROR}")

    # nº de modos por defecto = nº de pisos
    n_modos = max(1, int(niveles if modos is None else min(modos, niveles)))

    os.makedirs("assets", exist_ok=True)

    try:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)  # 2D, 3 DOF por nodo
    except Exception as e:
        return None, None, f"Error iniciando modelo: {e}"

    # Nodos: base empotrada (fix 1,1,1). Masa en nodos de niveles superiores.
    nodos = {}
    tag = 1
    for i in range(niveles + 1):         # 0..niveles
        y = i * h_nivel
        for j in range(panos + 1):       # 0..panos
            x = j * luz_viga            # << geometría según luz
            if i == 0:
                ops.node(tag, x, y)
                ops.fix(tag, 1, 1, 1)
            else:
                ops.node(tag, x, y, "-mass", m_nodal, m_nodal, 0.0)
            nodos[(i, j)] = tag
            tag += 1

    # Transformación geométrica y elementos
    ops.geomTransf("Linear", 1)

    # Columnas
    eid = 1000
    for i in range(niveles):
        for j in range(panos + 1):
            ni = nodos[(i, j)]
            nj = nodos[(i + 1, j)]
            ops.element("elasticBeamColumn", eid, ni, nj, A_col, E, Iz_col, 1, "-mass", 0.0)
            eid += 1

    # Vigas
    for i in range(1, niveles + 1):
        for j in range(panos):
            ni = nodos[(i, j)]
            nj = nodos[(i, j + 1)]
            ops.element("elasticBeamColumn", eid, ni, nj, A_viga, E, Iz_viga, 1, "-mass", 0.0)
            eid += 1

    # Autovalores y periodos
    try:
        lambdas = np.array(ops.eigen("-genBandArpack", n_modos), dtype=float)
        lambdas = np.where(lambdas > 0, lambdas, np.nan)
        omega = np.sqrt(lambdas)
        T = 2*np.pi/omega
    except Exception as e:
        return None, None, f"Error en eigen: {e}"

    # Formas modales (si hay opsvis)
    rutas = []
    if OPSVIS_OK:
        fmt_defo = {"color": "b", "linestyle": "-", "linewidth": 2}
        fmt_und  = {"color": "g", "linestyle": "--", "linewidth": 0.7}
        for k in range(n_modos):
            try:
                opsv.plot_mode_shape(k+1, endDispFlag=0, fmt_undefo=fmt_und, fmt_defo=fmt_defo)
                plt.title(f"$T_{k+1}$: {T[k]:.4f} s", fontweight="bold")
                ruta = f"assets/modo_{k+1}.png"
                plt.savefig(ruta, dpi=160, bbox_inches="tight")
                plt.close()
                rutas.append(ruta)
            except Exception:
                plt.close()
                rutas.append("")
    return T, rutas, None

# -----------------------------------------------------------------------------
# UI Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pórticos 2D – Modal (OpenSeesPy)", layout="wide")
st.title("Pórticos 2D – Análisis Modal (OpenSeesPy)")
st.caption("Unidades: m – kg – s. E en Pa (N/m²). Masas en kg.")

with st.sidebar:
    st.header("Parámetros de geometría")
    niveles = st.number_input("Niveles (pisos)", min_value=1, value=3, step=1)
    panos   = st.number_input("Paños", min_value=1, value=2, step=1)
    h_nivel = st.number_input("Altura por nivel h [m]", min_value=0.0, value=3.0, step=0.1, format="%.2f")
    luz     = st.number_input("Luz de viga L [m] (separación entre ejes de columnas)",
                              min_value=0.0, value=6.0, step=0.1, format="%.2f")

    st.header("Propiedades y masa (m–kg–s)")
    m_nodal = st.number_input("Masa nodal [kg]", min_value=0.0, value=1000.0, step=10.0, format="%.1f")
    E       = st.number_input("Módulo E [Pa]", min_value=1e6, value=2.0e10, step=1e9, format="%.0f")
    A_col   = st.number_input("A columna [m²]", min_value=0.0, value=0.09, step=0.001, format="%.3f")
    Iz_col  = st.number_input("Iz columna [m⁴]", min_value=0.0, value=0.000675, step=0.000001, format="%.6f")
    A_viga  = st.number_input("A viga [m²]", min_value=0.0, value=0.09, step=0.001, format="%.3f")
    Iz_viga = st.number_input("Iz viga [m⁴]", min_value=0.0, value=1.0e12, step=1.0e10, format="%.0f")

    st.header("Modos")
    n_modos_ui = st.number_input("Nº de modos a calcular (≤ niveles)",
                                 min_value=1, max_value=int(niveles), value=int(niveles), step=1)

    st.markdown("---")
    if OPS_OK:
        st.success("OpenSeesPy ✓ disponible")
    else:
        st.error("OpenSeesPy no está disponible en este entorno.")
        st.stop()

    if not OPSVIS_OK:
        st.info("OpsVis no está disponible: se omitirán imágenes de modos.")

    lanzar = st.button("Calcular")

# Resultados
if lanzar:
    with st.spinner("Ejecutando análisis modal..."):
        T, rutas, err = armar_y_analizar(
            int(niveles), int(panos),
            float(h_nivel), float(luz),
            float(m_nodal),
            float(E), float(A_col), float(Iz_col), float(A_viga), float(Iz_viga),
            modos=int(n_modos_ui)
        )

    if err:
        st.error(err)
    else:
        st.subheader("Períodos")
        for i, ti in enumerate(T, start=1):
            st.write(f"**T{i} = {ti:.4f} s**")

        st.subheader("Formas Modales")
        if OPSVIS_OK and any(rutas):
            # grilla 2 columnas
            cols = st.columns(2)
            for i, ruta in enumerate(rutas):
                if ruta and os.path.exists(ruta):
                    with cols[i % 2]:
                        st.image(ruta, caption=f"Modo {i+1}", use_column_width=True)
        else:
            st.info("No se muestran imágenes (opsvis no disponible).")
else:
    st.info("Ajusta los parámetros en la barra lateral y presiona **Calcular**.")
