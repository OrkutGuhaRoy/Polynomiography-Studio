import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from typing import List

st.set_page_config(page_title="Polynomiography Studio", layout="wide")
st.title("🎨 Polynomiography Studio — Exploring Newton Fractals with The Z³ Collectives")
st.caption("An interactive environment for visualizing and comparing polynomial root-finding dynamics through multi-function fractal generation, custom color palettes, and 3D analytical views.")

# -------------------------
# Sidebar: Mode + controls
# -------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio("Mode", ["Circular Roots (auto)", "Custom f(z)"])
st.sidebar.markdown("---")

# -------------------------
# Dynamic function list + palette control
# -------------------------
if "functions" not in st.session_state:
    st.session_state.functions = ["z**3 - 1"]
if "palettes" not in st.session_state:
    st.session_state.palettes = ["viridis"]

if mode == "Circular Roots (auto)":
    n_roots = st.sidebar.slider("Number of roots", 2, 12, 3)
    functions = [f"circular_{n_roots}"]
else:
    st.sidebar.markdown("### Custom Functions")
    funcs = st.session_state.functions
    pals = st.session_state.palettes

    new_funcs, new_pals = [], []
    available_palettes = [
    # Perceptually Uniform Sequential
    "viridis", "plasma", "inferno", "magma", "cividis", "rocket", "flare", "crest", "mako",

    # Sequential (Classic)
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", 
    "YlGnBu", "PuBuGn", "BuGn", "YlGn",

    # Diverging
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu",
    "RdYlGn", "Spectral", "coolwarm", "seismic", "twilight", "twilight_shifted",

    # Cyclic / Modern
    "turbo", "hsv", "cubehelix", "ocean", "icefire", "vlag", "balance", "tempo",

    # Qualitative / Categorical
    "tab10", "tab20", "tab20b", "tab20c",
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",

    # Misc / Specialty
    "terrain", "gist_earth", "gist_heat", "hot", "cool", "spring", "summer", "autumn", "winter",
    "Wistia", "CMRmap", "flag", "prism", "nipy_spectral"
]


    for idx, fexpr in enumerate(funcs):
        cols = st.sidebar.columns([0.8, 0.2])
        new_val = cols[0].text_input(f"f{idx+1}(z)", value=fexpr, key=f"func_input_{idx}")
        # Palette picker per function
        pal = st.sidebar.selectbox(f"🎨 Palette for f{idx+1}", available_palettes,
                                   index=available_palettes.index(pals[idx]) if pals[idx] in available_palettes else 2,
                                   key=f"palette_{idx}")
        new_funcs.append(new_val)
        new_pals.append(pal)

        c2 = st.sidebar.columns([0.4, 0.3, 0.3])
        if c2[0].button("↑", key=f"up_{idx}") and idx > 0:
            funcs[idx - 1], funcs[idx] = funcs[idx], funcs[idx - 1]
            pals[idx - 1], pals[idx] = pals[idx], pals[idx - 1]
            st.rerun()
        if c2[1].button("×", key=f"del_{idx}"):
            funcs[idx] = None
            pals[idx] = None

    funcs = [f for f in funcs if f is not None]
    pals = [p for p in pals if p is not None]

    for i, (fexpr, pal) in enumerate(zip(new_funcs, new_pals)):
        if i < len(funcs):
            funcs[i] = fexpr
            pals[i] = pal
        else:
            funcs.append(fexpr)
            pals.append(pal)

    st.session_state.functions = funcs
    st.session_state.palettes = pals

    if st.sidebar.button("➕ Add function", key="add_func"):
        st.session_state.functions.append("z**3 - 1")
        st.session_state.palettes.append("viridis")
        st.rerun()

    functions = st.session_state.functions[:]
    palettes_custom = st.session_state.palettes[:]

# -------------------------
# General parameters
# -------------------------
st.sidebar.markdown("---")
max_iter = st.sidebar.slider("Max iterations", 10, 400, 80)
tolerance = st.sidebar.number_input("Tolerance", 1e-8, 1e-2, 1e-5, format="%.1e")
zoom = st.sidebar.slider("Zoom (half-width)", 0.05, 5.0, 1.5, step=0.05)
center_x = st.sidebar.number_input("Center Re", value=0.0, format="%.6f")
center_y = st.sidebar.number_input("Center Im", value=0.0, format="%.6f")
res = st.sidebar.slider("Resolution (pixels)", 150, 900, 400, step=50)

if mode == "Circular Roots (auto)":
    global_palette = st.sidebar.selectbox("Color palette", ["tab10", "inferno", "plasma", "viridis", "magma", "cividis", "twilight"])

view_mode = st.sidebar.selectbox("View mode", ["2D", "3D (iterations)", "3D (|f(z)|)"])
st.sidebar.markdown("---")

# -------------------------
# Multi-function view options
# -------------------------
if mode == "Custom f(z)" and len(functions) > 1:
    multi_view = st.sidebar.radio("Multi-function 2D view", ["Side-by-side", "Overlay"])
    if multi_view == "Overlay":
        overlay_style = st.sidebar.radio("Overlay style", ["Distinct colors (per function)", "Blended intensity (mix)"])
else:
    multi_view, overlay_style = None, None

if mode == "Custom f(z)":
    sel_idx = st.sidebar.selectbox("3D: choose function", list(range(len(functions))),
                                   format_func=lambda i: f"f{i+1}(z) = {functions[i]}")
else:
    sel_idx = 0

# -------------------------
# Grid
# -------------------------
x = np.linspace(center_x - zoom, center_x + zoom, res)
y = np.linspace(center_y - zoom, center_y + zoom, res)
X, Y = np.meshgrid(x, y)
Z0 = X + 1j * Y

# -------------------------
# Function builder
# -------------------------
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
            val = np.zeros_like(z, dtype=complex)
            for i in range(len(roots)):
                prod = np.ones_like(z, dtype=complex)
                for j, r in enumerate(roots):
                    if i != j:
                        prod *= (z - r)
                val += prod
            return val
        desc = f"circular roots n={n}"
        return f, fprime, desc
    else:
        expr = spec
        allowed = {
            "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "abs": np.abs, "real": np.real, "imag": np.imag, "conj": np.conjugate
        }
        code = compile(expr, "<user_expr>", "eval")
        def f(z): return eval(code, allowed, {"z": z})
        def fprime(z):
            h = 1e-6
            return (f(z + h) - f(z - h)) / (2 * h)
        return f, fprime, f"f(z) = {expr}"

# -------------------------
# Newton iteration (vectorized)
# -------------------------
def newton_vectorized(f, fp, Z, max_iter, tol):
    Z = Z.copy()
    iters = np.full(Z.shape, max_iter, dtype=np.int32)
    conv = np.zeros(Z.shape, dtype=bool)
    for k in range(max_iter):
        fz = f(Z)
        fz_p = fp(Z)
        fz_p = np.where(np.abs(fz_p) < 1e-14, 1e-14, fz_p)
        Znext = Z - fz / fz_p
        step = np.abs(Znext - Z)
        newly = (step < tol) & (~conv)
        iters[newly] = k
        conv |= newly
        Z = Znext
        if conv.all(): break
    return Z, iters, conv

# -------------------------
# Compute
# -------------------------
results = []
for idx, spec in enumerate(functions):
    f, fp, desc = build_functions_for_spec(spec)
    final_Z, iters, conv = newton_vectorized(f, fp, Z0, max_iter, tolerance)
    if mode == "Circular Roots (auto)":
        n = int(spec.split("_")[1])
        roots_local = np.array([np.exp(2j * np.pi * k / n) for k in range(n)])
        root_index = np.full(final_Z.shape, -1)
        for i, r in enumerate(roots_local):
            root_index[np.abs(final_Z - r) < tolerance] = i
    else:
        root_index = np.where(conv, 1, -1)
    results.append({"desc": desc, "f": f, "final_Z": final_Z, "iters": iters,
                    "conv": conv, "root_index": root_index,
                    "palette": palettes_custom[idx] if mode == "Custom f(z)" else global_palette})

# -------------------------
# Visualization
# -------------------------
if view_mode == "2D":
    if mode == "Custom f(z)" and len(results) > 1 and multi_view == "Side-by-side":
        ncols = len(results)
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6))
        if ncols == 1: axes = [axes]
        for ax, res in zip(axes, results):
            ax.imshow(res["iters"], cmap=res["palette"], extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
            ax.set_title(res["desc"])
            ax.axis("off")
        st.pyplot(fig)

    elif mode == "Custom f(z)" and len(results) > 1 and multi_view == "Overlay":
        H, W = Z0.shape
        img = np.zeros((H, W, 3), dtype=float)

        for res in results:
            cmap = plt.get_cmap(res["palette"])
            conv = res["conv"]
            iters = res["iters"].astype(float)
            # Normalize iterations to [0,1]
            norm = np.clip(iters / max_iter, 0, 1)
            color_layer = cmap(1 - norm)[..., :3]  # brighter for faster convergence
            # Apply only where it converged
            color_layer *= conv[..., None]
            # Additively blend (normalize later)
            img += color_layer

        # Normalize final image for vivid look
        img /= np.maximum(img.max(), 1e-8)
        img = np.clip(img, 0, 1)

        fig, ax = plt.subplots(figsize=(7,7))
        ax.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
        ax.set_title("Overlay — Additive Blend (clean mode)")
        ax.axis("off")
        st.pyplot(fig)

    else:
        res0 = results[0]
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(res0["iters"], cmap=res0["palette"], extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
        ax.set_title(res0["desc"])
        ax.axis("off")
        st.pyplot(fig)

# -------------------------
# 3D view (one function)
# -------------------------
if view_mode.startswith("3D"):
    chosen = results[sel_idx]
    if view_mode == "3D (iterations)":
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=chosen["iters"], colorscale=chosen["palette"])])
        fig.update_layout(scene=dict(xaxis_title="Re(z)", yaxis_title="Im(z)", zaxis_title="Iterations"),
                          title=f"3D Iterations — {chosen['desc']}")
    else:
        absf = np.abs(chosen["f"](Z0))
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=absf, colorscale=chosen["palette"])])
        fig.update_layout(scene=dict(xaxis_title="Re(z)", yaxis_title="Im(z)", zaxis_title="|f(z)|"),
                          title=f"3D |f(z)| — {chosen['desc']}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.success("✅ Done — multi-function fractal ready")
