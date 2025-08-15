# -*- coding: utf-8 -*-
# Pórticos 2D – Análisis Modal con ANIMACIONES de modos (OpenSeesPy)
# Unidades: m–kg–s (E en Pa). Comentarios e interfaz en español.

import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")                     # necesario en entornos headless
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

# -------------------------------------------------------------------
# Importaciones tolerantes
# -------------------------------------------------------------------
OPS_OK, OPS_IMPORT_ERROR = True, ""
try:
    import openseespy.opensees as ops
except Exception as e:
    OPS_OK, OPS_IMPORT_ERROR = False, str(e)

# -------------------------------------------------------------------
# Utilidades para geometría, eigenvectores y animación
# -------------------------------------------------------------------
def build_portico(niveles:int, panos:int, h:float, L:float,
                  m_nodal:float, E:float, A_col:float, Iz_col:float,
                  A_viga:float, Iz_viga:float):
    """
    Crea el modelo en OpenSeesPy y retorna:
    - tags_nodos: lista ordenada de tags de nodos
    - coords: dict {tag: (x,y)}
    - elems: lista de pares (ni, nj) con la conectividad (columnas y vigas)
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    coords = {}
    nodos = {}
    tag = 1
    for i in range(niveles + 1):
        y = i * h
        for j in range(panos + 1):
            x = j * L                              # << geometría según luz
            if i == 0:
                ops.node(tag, x, y)
                ops.fix(tag, 1, 1, 1)
            else:
                ops.node(tag, x, y, "-mass", m_nodal, m_nodal, 0.0)
            coords[tag] = (x, y)
            nodos[(i, j)] = tag
            tag += 1

    ops.geomTransf("Linear", 1)
    elems = []

    # Columnas
    eid = 1000
    for i in range(niveles):
        for j in range(panos + 1):
            ni = nodos[(i, j)]
            nj = nodos[(i + 1, j)]
            ops.element("elasticBeamColumn", eid, ni, nj, A_col, E, Iz_col, 1, "-mass", 0.0)
            elems.append((ni, nj))
            eid += 1

    # Vigas
    for i in range(1, niveles + 1):
        for j in range(panos):
            ni = nodos[(i, j)]
            nj = nodos[(i, j + 1)]
            ops.element("elasticBeamColumn", eid, ni, nj, A_viga, E, Iz_viga, 1, "-mass", 0.0)
            elems.append((ni, nj))
            eid += 1

    tags_nodos = sorted(coords.keys())
    return tags_nodos, coords, elems


def calc_modal(niveles:int, n_modos:int):
    """
    Calcula autovalores y retorna periodos T y lista de eigenvectores por modo:
    - T : np.ndarray [s]
    - modos: list[ dict{tag: (ux, uy)} ]  (desplazamientos modales por nodo)
    """
    n_modos = max(1, min(n_modos, niveles))
    lambdas = np.array(ops.eigen("-genBandArpack", n_modos), dtype=float)
    lambdas = np.where(lambdas > 0, lambdas, np.nan)
    omega = np.sqrt(lambdas)
    T = 2*np.pi/omega

    # extraer eigenvectores por nodo/dof desde OpenSees
    modos = []
    # obtén todos los nodos definidos actualmente
    all_nodes = ops.getNodeTags()
    for k in range(1, n_modos + 1):
        vec = {}
        for tag in all_nodes:
            ux = ops.nodeEigenvector(tag, k, 1)
            uy = ops.nodeEigenvector(tag, k, 2)
            vec[tag] = (ux, uy)
        modos.append(vec)
    return T, modos


def make_segments(coords_dict, elems):
    """Convierte coords y conectividad a segmentos [[(x1,y1),(x2,y2)], ...]."""
    segs = []
    for ni, nj in elems:
        x1, y1 = coords_dict[ni]
        x2, y2 = coords_dict[nj]
        segs.append([(x1, y1), (x2, y2)])
    return segs


def animate_mode(coords_dict, elems, evec_dict, scale_geom:float,
                 out_path:str, frames:int=40, fps:int=20):
    """
    Genera un GIF del modo:
    - coords_dict: {tag:(x,y)} coordenadas indeformadas
    - elems: [(ni, nj)]
    - evec_dict: {tag:(ux,uy)} eigenvector modal
    - scale_geom: escala geométrica (amplitud máxima ~ scale_geom * dimensión característica)
    - out_path: archivo GIF destino
    """
    # dimensión característica (para normalizar la amplitud visual)
    xs = [xy[0] for xy in coords_dict.values()]
    ys = [xy[1] for xy in coords_dict.values()]
    Lx = max(xs) - min(xs) if xs else 1.0
    Ly = max(ys) - min(ys) if ys else 1.0
    dim = max(Lx, Ly, 1.0)                 # p.ej. 6–12 m típicos
    amp = scale_geom * dim                 # amplitud gráfica

    # normalizar eigenvector a desplazamiento máximo = 1
    mag = max(np.hypot(*evec_dict[tag]) for tag in evec_dict) or 1.0

    # datos base
    segs0 = make_segments(coords_dict, elems)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
    lc = LineCollection(segs0, colors="b", linewidths=2)
    ax.add_collection(lc)

    # malla indeformada discontinua
    lc0 = LineCollection(segs0, colors="g", linewidths=0.7, linestyles="--")
    ax.add_collection(lc0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(xs)-0.2*dim, max(xs)+0.2*dim)
    ax.set_ylim(min(ys)-0.2*dim, max(ys)+0.2*dim)
    ax.set_xticks([]); ax.set_yticks([])

    def frame_segments(phi):
        """Retorna segmentos deformados para fase 'phi' (0..2π)."""
        coords_def = {}
        for tag, (x, y) in coords_dict.items():
            ux, uy = evec_dict.get(tag, (0.0, 0.0))
            coords_def[tag] = (x + amp * (ux/mag) * np.sin(phi),
                               y + amp * (uy/mag) * np.sin(phi))
        segs = []
        for ni, nj in elems:
            segs.append([coords_def[ni], coords_def[nj]])
        return segs

    def init():
        lc.set_segments(segs0)
        return (lc,)

    def update(f):
        phi = 2*np.pi * f/frames
        lc.set_segments(frame_segments(phi))
        return (lc,)

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=frames, interval=1000/fps, blit=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)
    finally:
        plt.close(fig)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Pórticos 2D – Modos animados (OpenSeesPy)", layout="wide")
st.title("Pórticos 2D – Análisis Modal (OpenSeesPy) con Animación")
st.caption("Unidades: m – kg – s. E en Pa (N/m²). Masas en kg. Las animaciones se guardan como GIF.")

with st.sidebar:
    st.header("Geometría")
    niveles = st.number_input("Niveles (pisos)", min_value=1, value=3, step=1)
    panos   = st.number_input("Paños", min_value=1, value=2, step=1)
    h       = st.number_input("Altura por nivel h [m]", min_value=0.0, value=3.0, step=0.1, format="%.2f")
    L       = st.number_input("Luz de viga L [m] (separación entre ejes de columnas)",
                              min_value=0.0, value=6.0, step=0.1, format="%.2f")

    st.header("Propiedades y masa")
    m_nodal = st.number_input("Masa nodal [kg]", min_value=0.0, value=1000.0, step=10.0, format="%.1f")
    E       = st.number_input("Módulo de elasticidad E [Pa]", min_value=1e6, value=2.0e10, step=1e9, format="%.0f")
    A_col   = st.number_input("A columna [m²]", min_value=0.0, value=0.09, step=0.001, format="%.3f")
    Iz_col  = st.number_input("Iz columna [m⁴]", min_value=0.0, value=0.000675, step=0.000001, format="%.6f")
    A_viga  = st.number_input("A viga [m²]", min_value=0.0, value=0.09, step=0.001, format="%.3f")
    Iz_viga = st.number_input("Iz viga [m⁴]", min_value=0.0, value=1.0e12, step=1.0e10, format="%.0f")

    st.header("Modos y animación")
    n_modos = st.number_input("Nº de modos (≤ niveles)", min_value=1, max_value=int(niveles),
                              value=int(niveles), step=1)
    escala = st.slider("Amplitud gráfica (fracción de la dimensión del pórtico)",
                       min_value=0.005, max_value=0.2, value=0.05, step=0.005)
    fps    = st.slider("FPS del GIF", min_value=5, max_value=30, value=20, step=1)
    frames = st.slider("Frames por ciclo", min_value=20, max_value=120, value=40, step=5)

    st.markdown("---")
    if OPS_OK:
        st.success("OpenSeesPy ✓ disponible")
    else:
        st.error("OpenSeesPy no está disponible.\n\n" + OPS_IMPORT_ERROR)
        st.stop()

    lanzar = st.button("Calcular y animar")

if lanzar:
    with st.spinner("Armando el modelo…"):
        tags, coords, elems = build_portico(int(niveles), int(panos), float(h), float(L),
                                            float(m_nodal), float(E),
                                            float(A_col), float(Iz_col), float(A_viga), float(Iz_viga))
    with st.spinner("Cálculo modal…"):
        T, modos = calc_modal(int(niveles), int(n_modos))

    # Mostrar períodos
    st.subheader("Períodos")
    for i, ti in enumerate(T, start=1):
        st.write(f"**T{i} = {ti:.4f} s**")

    # Generar y mostrar GIFs
    st.subheader("Animaciones de modos")
    gif_paths = []
    with st.spinner("Generando GIFs…"):
        for k, evec in enumerate(modos, start=1):
            out_path = f"assets/modo_{k}.gif"
            animate_mode(coords, elems, evec, float(escala), out_path,
                         frames=int(frames), fps=int(fps))
            gif_paths.append(out_path)

    cols = st.columns(2)
    for i, pth in enumerate(gif_paths):
        if os.path.exists(pth):
            with cols[i % 2]:
                st.image(pth, caption=f"Modo {i+1}", use_column_width=True)
else:
    st.info("Ajusta los parámetros en la barra lateral y presiona **Calcular y animar**.")
