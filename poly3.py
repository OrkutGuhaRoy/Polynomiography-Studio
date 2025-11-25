import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Tuple
import math
import warnings
import matplotlib.cm as cm
import matplotlib as mpl # Added for robust colormap access

warnings.filterwarnings("ignore")

# =============================================================
#  COLOR FIX: Updated to handle Matplotlib deprecations
# =============================================================
def mpl_to_plotly_cmap(name, n=256):
    try:
        # Try modern matplotlib API (colormaps registry)
        cmap = mpl.colormaps[name]
    except (AttributeError, KeyError):
        try:
            # Fallback for older matplotlib or via pyplot
            cmap = plt.get_cmap(name)
        except:
            # Final fallback
            cmap = plt.get_cmap("viridis")
            
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(int)
    return [[i/(n - 1), f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(colors)]
# =============================================================


st.set_page_config(page_title="Polynomiography Studio", layout="wide")
st.title("🎨 Polynomiography Studio — Advanced Dynamics")
st.caption("Exploring Newton Fractals, Halley's Method, and Quadtree Refinement.")

# -------------------------
# Sidebar: Mode + controls
# -------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio("Mode", ["Circular Roots (auto)", "Custom f(z)"])
st.sidebar.markdown("---")

# -------------------------
# Global Algorithm (for circular mode)
# -------------------------
st.sidebar.subheader("🧮 Algorithms & Methods (Global)")
global_algo_method = st.sidebar.selectbox(
    "Iteration Scheme (global, used for Circular Roots)",
    ["Newton", "Halley", "Relaxed Newton (α)", "Hybrid (Adaptive)"],
    index=0
)
global_alpha = 1.0
if global_algo_method == "Relaxed Newton (α)":
    global_alpha = st.sidebar.slider("Global relaxation parameter (α)", 0.1, 2.0, 1.0, 0.1)

use_adaptive = st.sidebar.checkbox("Adaptive Step Control (global)", value=(global_algo_method == "Hybrid (Adaptive)"))
use_refinement = st.sidebar.checkbox(
    "Quadtree Boundary Refinement (global)",
    value=True,
    help="Computes low-res first, then refines edges. Faster for large images."
)

st.sidebar.markdown("---")

# -------------------------
# Session state defaults
# -------------------------
if "functions" not in st.session_state:
    st.session_state.functions = ["z**3 + 1"]
if "palettes" not in st.session_state:
    st.session_state.palettes = ["viridis"]
if "algos" not in st.session_state:
    st.session_state.algos = ["Newton"]
if "alphas" not in st.session_state:
    st.session_state.alphas = [1.0]

# -------------------------
# Dynamic function list + palette control (improved UI)
# -------------------------
available_palettes = [
    "viridis", "plasma", "inferno", "magma", "cividis", "twilight", "turbo",
    "Greys", "Blues", "Reds", "Spectral", "coolwarm", "tab10", "ocean"
]

if mode == "Circular Roots (auto)":
    n_roots = st.sidebar.slider("Number of roots", 2, 12, 3)
    functions = [f"circular_{n_roots}"]
else:
    st.sidebar.markdown("### Custom Functions")
    funcs = st.session_state.functions[:]
    pals = st.session_state.palettes[:]
    algos = st.session_state.algos[:]
    alphas = st.session_state.alphas[:]

    n_items = max(len(funcs), len(pals), len(algos), len(alphas))
    while len(funcs) < n_items:
        funcs.append("z**3 + 1")
    while len(pals) < n_items:
        pals.append("viridis")
    while len(algos) < n_items:
        algos.append("Newton")
    while len(alphas) < n_items:
        alphas.append(1.0)

    new_funcs, new_pals, new_algos, new_alphas = [], [], [], []
    for idx, fexpr in enumerate(funcs):
        with st.sidebar.expander(f"f{idx+1}(z): {fexpr}", expanded=False):
            new_val = st.text_input(f"Expression f{idx+1}(z)", value=fexpr, key=f"func_input_{idx}")
            pal = st.selectbox(
                f"Palette (f{idx+1})",
                available_palettes,
                index=available_palettes.index(pals[idx]) if pals[idx] in available_palettes else 0,
                key=f"palette_{idx}"
            )
            algo = st.selectbox(
                f"Method for f{idx+1}",
                ["Newton", "Halley", "Relaxed Newton (α)", "Hybrid (Adaptive)"],
                index=["Newton", "Halley", "Relaxed Newton (α)", "Hybrid (Adaptive)"].index(algos[idx]) if algos[idx] in ["Newton", "Halley", "Relaxed Newton (α)", "Hybrid (Adaptive)"] else 0,
                key=f"method_{idx}"
            )
            if algo == "Relaxed Newton (α)":
                alpha_local = st.slider(f"α (f{idx+1})", 0.1, 2.0, float(alphas[idx]), 0.1, key=f"alpha_{idx}")
            else:
                alpha_local = float(alphas[idx]) if alphas[idx] is not None else 1.0

            c1, c2, c3 = st.columns([0.6, 0.2, 0.2])
            if c2.button("Delete", key=f"del_{idx}"):
                st.session_state._delete_index = idx
            if c3.button("Duplicate", key=f"dup_{idx}"):
                new_funcs.append(new_val)
                new_pals.append(pal)
                new_algos.append(algo)
                new_alphas.append(alpha_local)
                new_funcs.append(new_val)
                new_pals.append(pal)
                new_algos.append(algo)
                new_alphas.append(alpha_local)
                continue

            if not (hasattr(st.session_state, "_delete_index") and st.session_state._delete_index == idx):
                new_funcs.append(new_val)
                new_pals.append(pal)
                new_algos.append(algo)
                new_alphas.append(alpha_local)
            else:
                del st.session_state._delete_index

    if st.sidebar.button("➕ Add function", key="add_func"):
        new_funcs.append("z**3 + 1")
        new_pals.append("viridis")
        new_algos.append("Newton")
        new_alphas.append(1.0)
        st.session_state.functions = new_funcs
        st.session_state.palettes = new_pals
        st.session_state.algos = new_algos
        st.session_state.alphas = new_alphas
        st.rerun()

    st.session_state.functions = new_funcs
    st.session_state.palettes = new_pals
    st.session_state.algos = new_algos
    st.session_state.alphas = new_alphas

    functions = st.session_state.functions[:]
    palettes_custom = st.session_state.palettes[:]

st.sidebar.markdown("---")

# -------------------------
# General parameters
# -------------------------
max_iter = int(st.sidebar.slider("Max iterations", 10, 500, 60))
tolerance = float(st.sidebar.number_input("Tolerance (step)", 1e-12, 1e-2, 1e-5, format="%.1e"))
zoom = float(st.sidebar.slider("Zoom (half-width)", 0.05, 5.0, 1.5, step=0.05))
center_x = float(st.sidebar.number_input("Center Re", value=0.0, format="%.6f"))
center_y = float(st.sidebar.number_input("Center Im", value=0.0, format="%.6f"))
res = int(st.sidebar.slider("Resolution (pixels)", 100, 1200, 400, step=50))

if mode == "Circular Roots (auto)":
    global_palette = st.sidebar.selectbox("Color palette", ["tab10", "inferno", "plasma", "viridis", "magma", "cividis", "twilight"])

view_mode = st.sidebar.selectbox("View mode", ["2D Basin", "2D Iterations", "Stability Norm A(z)", "3D (iterations)", "3D (|f(z)|)"])
st.sidebar.markdown("---")

# -------------------------
# Multi-function view options
# -------------------------
multi_view = None
overlay_style = None
if mode == "Custom f(z)" and len(functions) > 1:
    multi_view = st.sidebar.radio("Multi-function 2D view", ["Side-by-side", "Overlay"])
    if multi_view == "Overlay":
        overlay_style = st.sidebar.radio("Overlay style", ["Distinct colors", "Blended intensity"])

if mode == "Custom f(z)":
    sel_idx = st.sidebar.selectbox("3D/Stability: choose function", list(range(len(functions))),
                                  format_func=lambda i: f"f{i+1}(z) = {functions[i]}")
else:
    sel_idx = 0

# -------------------------
# Function builder
# -------------------------
def safe_compile_expr(expr: str):
    allowed = {
        "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
        "real": np.real, "imag": np.imag, "conj": np.conjugate, "pi": math.pi,
        "e": math.e
    }
    blacklist = ["import", "__", "os.", "sys.", "subprocess", "open(", "eval(", "exec("]
    for b in blacklist:
        if b in expr:
            raise ValueError(f"Disallowed token in expression: {b}")

    code = compile(expr, "<user_expr>", "eval")
    def f(z):
        return eval(code, allowed, {"z": z})
    return f

def build_functions_for_spec(spec):
    if isinstance(spec, str) and spec.startswith("circular_"):
        n = int(spec.split("_")[1])
        roots = np.array([np.exp(2j * np.pi * k / n) for k in range(n)])
        def f(z):
            val = np.ones_like(z, dtype=complex)
            for r in roots:
                val *= (z - r)
            return val
        def fprime(z):
            h = 1e-5
            return (f(z + h) - f(z - h)) / (2 * h)
        def fprime2(z):
            h = 1e-5
            return (f(z + h) - 2 * f(z) + f(z - h)) / (h ** 2)
        desc = f"circular roots n={n}"
        return f, fprime, fprime2, desc
    else:
        expr = spec
        try:
            f_callable = safe_compile_expr(expr)
        except Exception as e:
            st.error(f"Error compiling expression '{expr}': {e}")
            def f_callable(z):
                return np.ones_like(z) * (1e6 + 0j)

        def f(z): return f_callable(z)

        def fprime(z):
            h = 1e-6
            return (f(z + h) - f(z - h)) / (2 * h)

        def fprime2(z):
            h = 1e-6
            return (f(z + h) - 2 * f(z) + f(z - h)) / (h ** 2)

        return f, fprime, fprime2, f"f(z) = {expr}"

# -------------------------
# Core Solver
# -------------------------
def fractal_solver(Z_init, f, fp, fpp, method, max_iter, tol, alpha, adaptive):
    Z = Z_init.copy()
    iters = np.full(Z.shape, max_iter, dtype=np.int32)
    conv = np.zeros(Z.shape, dtype=bool)

    stability = np.zeros(Z.shape, dtype=float)

    for k in range(max_iter):
        mask = ~conv
        if not np.any(mask):
            break

        z_curr = Z[mask]
        fz = f(z_curr)
        fpz = fp(z_curr)

        fpz = np.where(np.abs(fpz) < 1e-12, 1e-12, fpz)

        if method == "Newton":
            step = fz / fpz
        elif method == "Halley":
            fppz = fpp(z_curr)
            denom = 2 * fpz ** 2 - fz * fppz
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            step = (2 * fz * fpz) / denom
        elif method == "Relaxed Newton (α)":
            step = alpha * (fz / fpz)
        elif method == "Hybrid (Adaptive)":
            step = fz / fpz
        else:
            step = fz / fpz

        if adaptive:
            step = np.where(np.abs(step) > 0.5, step * 0.5, step)
            step = np.where(np.abs(fz) > 10.0, step * 0.5, step)

        Z[mask] -= step

        diff = np.abs(step)
        stability[mask] += diff

        done_now = diff < tol

        iters[mask] = np.where(done_now, k, iters[mask])
        conv[mask] |= done_now

    return Z, iters, conv, stability

# -------------------------
# Quadtree refinement
# -------------------------
def compute_with_refinement(X, Y, f, fp, fpp, method, max_iter, tol, alpha, adaptive, refinement):
    if not refinement:
        Z0 = X + 1j * Y
        Z_final, iters, conv, stab = fractal_solver(Z0, f, fp, fpp, method, max_iter, tol, alpha, adaptive)
        return Z_final, iters, conv, stab

    step = 4
    X_coarse = X[::step, ::step]
    Y_coarse = Y[::step, ::step]
    Z_coarse = X_coarse + 1j * Y_coarse

    Z_c, iters_c, conv_c, stab_c = fractal_solver(Z_coarse, f, fp, fpp, method, max_iter, tol, alpha, adaptive)

    Z_full = np.kron(Z_c, np.ones((step, step)))
    iters_full = np.kron(iters_c, np.ones((step, step), dtype=int))
    conv_full = np.kron(conv_c, np.ones((step, step), dtype=bool))
    stab_full = np.kron(stab_c, np.ones((step, step)))

    h, w = X.shape
    Z_full = Z_full[:h, :w]
    iters_full = iters_full[:h, :w]
    conv_full = conv_full[:h, :w]
    stab_full = stab_full[:h, :w]

    dy, dx = np.gradient(iters_full)
    edge_mask = (np.abs(dx) > 0) | (np.abs(dy) > 0) | (~conv_full)

    padded = np.pad(edge_mask, 1, mode='constant', constant_values=False)
    dilated = (edge_mask | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])

    if np.any(dilated):
        Z_grid = X + 1j * Y
        Z_active_init = Z_grid[dilated]

        Z_res, iters_res, conv_res, stab_res = fractal_solver(Z_active_init, f, fp, fpp, method, max_iter, tol, alpha, adaptive)

        Z_full[dilated] = Z_res
        iters_full[dilated] = iters_res
        conv_full[dilated] = conv_res
        stab_full[dilated] = stab_res

    return Z_full, iters_full, conv_full, stab_full

# -------------------------
# Main Compute Loop
# -------------------------
x = np.linspace(center_x - zoom, center_x + zoom, res)
y = np.linspace(center_y - zoom, center_y + zoom, res)
X, Y = np.meshgrid(x, y)

results = []
for idx, spec in enumerate(functions):
    f, fp, fpp, desc = build_functions_for_spec(spec)

    if mode == "Circular Roots (auto)":
        method_local = global_algo_method
        alpha_local = global_alpha
    else:
        method_local = st.session_state.algos[idx] if idx < len(st.session_state.algos) else "Newton"
        alpha_local = float(st.session_state.alphas[idx]) if idx < len(st.session_state.alphas) else 1.0

    final_Z, iters, conv, stability = compute_with_refinement(
        X, Y, f, fp, fpp,
        method_local, max_iter, tolerance, alpha_local, use_adaptive, use_refinement
    )

    if mode == "Circular Roots (auto)":
        n = int(spec.split("_")[1])
        roots_local = np.array([np.exp(2j * np.pi * k / n) for k in range(n)])
        root_index = np.full(final_Z.shape, -1)
        for i, r in enumerate(roots_local):
            root_index[np.abs(final_Z - r) < 0.1] = i
        palette = global_palette
    else:
        root_index = np.angle(final_Z)
        palette = palettes_custom[idx] if idx < len(palettes_custom) else "viridis"

    results.append({
        "desc": desc,
        "f": f,
        "final_Z": final_Z,
        "iters": iters,
        "conv": conv,
        "root_index": root_index,
        "stability": stability,
        "palette": palette,
        "method": method_local,
        "alpha": alpha_local
    })

# -------------------------
# Visualization
# -------------------------
st.markdown(f"**Status:** Rendered {res}x{res} px | Refinement: {'ON' if use_refinement else 'OFF'} | View: {view_mode}")

if view_mode.startswith("2D"):
    is_basin = (view_mode == "2D Basin")

    if mode == "Custom f(z)" and len(results) > 1 and multi_view == "Side-by-side":
        ncols = len(results)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axes = [axes]
        for ax, res_data in zip(axes, results):
            data = res_data["root_index"] if is_basin else res_data["iters"]

            if is_basin and mode == "Custom f(z)":
                ax.imshow(data, cmap="hsv", extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
            else:
                ax.imshow(data, cmap=res_data["palette"], extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")

            ax.set_title(f"{res_data['desc']} — {res_data['method']}")
            ax.axis("off")
        st.pyplot(fig)

    elif mode == "Custom f(z)" and len(results) > 1 and multi_view == "Overlay":
        H, W = X.shape
        img = np.zeros((H, W, 3), dtype=float)

        for res_data in results:
            # Replaced deprecated cm.get_cmap calls with plt.get_cmap
            cmap = plt.get_cmap(res_data["palette"])
            conv_mask = res_data["conv"]
            val_data = res_data["iters"].astype(float)
            norm = np.clip(val_data / max_iter, 0, 1)

            color_layer = cmap(1 - norm)[..., :3]
            color_layer *= conv_mask[..., None]

            img += color_layer

        img /= np.maximum(img.max(), 1e-8)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.clip(img, 0, 1), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
        ax.axis("off")
        st.pyplot(fig)

    else:
        res0 = results[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        data = res0["root_index"] if is_basin else res0["iters"]

        if is_basin and mode == "Custom f(z)":
            im = ax.imshow(data, cmap="twilight", extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
        else:
            im = ax.imshow(data, cmap=res0["palette"], extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")

        ax.set_title(f"{res0['desc']} - {view_mode} — {res0['method']}")
        ax.axis("off")
        st.pyplot(fig)

elif view_mode == "Stability Norm A(z)":
    chosen = results[sel_idx]

    ### COLOR FIX
    cs = mpl_to_plotly_cmap(chosen["palette"])

    fig = go.Figure(data=[go.Heatmap(z=chosen["stability"], colorscale=cs)])
    fig.update_layout(title=f"Stability Norm Map (Cumulative Step) — {chosen['desc']} — {chosen['method']}")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode.startswith("3D"):
    chosen = results[sel_idx]

    ### COLOR FIX
    cs = mpl_to_plotly_cmap(chosen["palette"])

    if view_mode == "3D (iterations)":
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=chosen["iters"], colorscale=cs)])
        fig.update_layout(scene=dict(xaxis_title="Re", yaxis_title="Im", zaxis_title="Iterations"),
                          title=f"3D Iterations — {chosen['desc']} — {chosen['method']}")
    else:
        absf = np.abs(chosen["f"](chosen["final_Z"]))
        absf = np.clip(absf, 0, 5)
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=absf, colorscale=cs)])
        fig.update_layout(scene=dict(xaxis_title="Re", yaxis_title="Im", zaxis_title="|f(z)|"),
                          title=f"3D |f(z)| Residual — {chosen['desc']} — {chosen['method']}")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Small footer
# -------------------------
st.markdown("---")
st.markdown(
    "Tips: Use **Relaxed Newton (α)** to slow convergence for delicate basins. "
    "Use **Hybrid (Adaptive)** with Adaptive Step Control for robustness. "
    "Add multiple custom functions and try the Overlay mode for artistic blends."
)
