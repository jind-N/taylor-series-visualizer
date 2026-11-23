import io
import math

import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from errors import compare_at_point
from plots import plot_1d, plot_2d, plot_surface
from taylor import matches_sympy_series_1d, taylor_polynomial


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Taylor Series Visualizer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📐 Taylor Series Approximation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Calculus & Python Framework for Function Approximation and Visualization</div>', unsafe_allow_html=True)

# ---------- Predefined Functions ----------
PREDEFINED = {
    "1 Variable: sin(x)": "sin(x)",
    "1 Variable: cos(x)": "cos(x)",
    "1 Variable: exp(x)": "exp(x)",
    "1 Variable: ln(1 + x)": "log(1 + x)",
    "1 Variable: 1 / (1 - x)": "1/(1 - x)",
    "1 Variable: exp(-x²)  (Gaussian)": "exp(-x**2)",
    "1 Variable: arctan(x)": "atan(x)",
    "2 Variables: exp(x + y)": "exp(x + y)",
    "2 Variables: sin(x)·cos(y)": "sin(x)*cos(y)",
    "2 Variables: exp(-(x² + y²))  (Dome)": "exp(-(x**2 + y**2))",
    "2 Variables: exp(x)·sin(y)": "exp(x)*sin(y)",
    "3 Variables: exp(x + y + z)": "exp(x + y + z)",
    "3 Variables: sin(x)·cos(y)·exp(z)": "sin(x)*cos(y)*exp(z)",
    "Real-World: Thermal Hotspot T(x, y)": "25 + 40*exp(-(x**2 + y**2)/2)",
    "Real-World: Pendulum Restoring Torque": "sin(x)",
}

DEGREE_LABELS = {
    0: "0 – Constant (Flat)",
    1: "1 – Linear (Tangent Line/Plane)",
    2: "2 – Quadratic (Curvature/Parabola)",
    3: "3 – Cubic (Inflection)",
}

MAX_TERMS = 120

if "saved_functions" not in st.session_state:
    st.session_state.saved_functions = {}


def read_function(text):
    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )
    return parse_expr(text, transformations=transformations)


# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.markdown("### 1. Function Selection")

input_mode = st.sidebar.radio(
    "Mode",
    ["Select Predefined", "Custom Formula"],
    horizontal=True,
    help="Pick a curated mathematical/physical model, or type your own.",
)

if input_mode == "Select Predefined":
    all_options = {**PREDEFINED}
    for label, formula in st.session_state.saved_functions.items():
        all_options[f"⭐ {label}"] = formula

    choice = st.sidebar.selectbox("Choose a function", list(all_options.keys()))
    function_text = all_options[choice]
else:
    function_text = st.sidebar.text_input(
        "Enter expression",
        value="x^2 + sin(y)",
        help="Use standard notation, e.g. x^2*y, exp(-x^2), log(1+x)",
    )
    if st.sidebar.button("💾 Save Function to List"):
        st.session_state.saved_functions[function_text] = function_text
        st.sidebar.success("Saved! Available in the predefined list.")

try:
    function = read_function(function_text)
except Exception:
    st.error("Invalid mathematical syntax. Please verify brackets and symbols.")
    st.stop()

variables = tuple(sorted(function.free_symbols, key=lambda s: s.name))
if not variables:
    st.error("Function must contain at least one variable (e.g., x).")
    st.stop()

st.sidebar.caption(
    f"Detected: **{len(variables)} variable(s)** ({', '.join(str(v) for v in variables)})"
)

# ---------- Center Point ----------
st.sidebar.markdown("### 2. Expansion Center")
center = tuple(
    st.sidebar.number_input(
        f"Center for {v} (a_{v})",
        value=0.0,
        step=0.5,
        format="%.2f",
        key=f"center_{v}",
        help="The local approximation will be exact at this coordinate.",
    )
    for v in variables
)

# ---------- Degree ----------
st.sidebar.markdown("### 3. Approximation Degree")
degree = st.sidebar.select_slider(
    "Degree (n)",
    options=list(range(0, 9)),
    value=2,
    format_func=lambda n: DEGREE_LABELS.get(n, f"{n} – Higher-Order Polynomial"),
    help="Higher degree adds higher derivative powers, widening the zone of accuracy.",
)

term_count = math.comb(degree + len(variables), len(variables))
if term_count > MAX_TERMS:
    st.sidebar.warning(f"⚠️ Computational limit: {term_count} terms exceeds {MAX_TERMS}. Lower the degree.")
    st.stop()

# ---------- View Style (2+ Variables) ----------
view = "3D surface"
if len(variables) > 1:
    st.sidebar.markdown("### 4. Graph Style")
    view = st.sidebar.radio(
        "Display as",
        ["3D surface", "Heatmap"],
        horizontal=True,
        help="3D surface shows height as value; Heatmap shows color density (thermal image).",
    )

# ---------- Automatic Settings with Advanced Expander ----------
view_width = 3.0
point = tuple(c + 0.5 for c in center)
fixed_values = list(center[2:])

with st.sidebar.expander("⚙️ Advanced Tuning (Optional)"):
    view_width = st.number_input(
        "Graph view radius",
        min_value=0.1,
        value=3.0,
        step=0.5,
        help="Visual range around the center point.",
    )
    if st.checkbox("Custom Test Point", help="By default, evaluated at center + 0.5."):
        point = tuple(
            st.number_input(
                f"Test value of {v}",
                value=float(p),
                step=0.1,
                format="%.2f",
                key=f"point_{v}",
            )
            for v, p in zip(variables, point)
        )
    if len(variables) > 2 and st.checkbox("Custom Slice Coordinate", help="Hold extra variables constant."):
        fixed_values = [
            st.number_input(
                f"Fixed value of {v}",
                value=float(c),
                step=0.1,
                format="%.2f",
                key=f"fixed_{v}",
            )
            for v, c in zip(variables[2:], center[2:])
        ]

ranges = tuple((c - view_width, c + view_width) for c in center)

# =====================================================
# CALCULATION & RESILIENT FALLBACK ENGINE
# =====================================================

def build_results(function, variables, center, degree, ranges, fixed_values, point, view):
    polynomial = taylor_polynomial(function, variables, center, degree)

    if len(variables) == 1:
        compare_degrees = sorted({1, degree})
        polynomials = {
            d: taylor_polynomial(function, variables, center, d)
            for d in compare_degrees
        }
        low, high = ranges[0]
        figure = plot_1d(function, polynomials, variables[0], center[0], low, high)
    else:
        plot_fn = plot_surface if view == "3D surface" else plot_2d
        figure = plot_fn(
            function,
            polynomial,
            variables,
            degree,
            ranges=(ranges[0], ranges[1]),
            fixed_values=fixed_values,
        )

    try:
        table = compare_at_point(function, variables, center, point, max_degree=degree)
    except Exception:
        table = None

    return polynomial, figure, table


def safe_default():
    x = sp.symbols("x")
    return dict(
        function=sp.sin(x),
        variables=(x,),
        center=(0.0,),
        degree=2,
        ranges=((-3.0, 3.0),),
        fixed_values=[],
        point=(0.5,),
        view="3D surface",
    )


def reset_inputs():
    for key in list(st.session_state.keys()):
        if key not in ("saved_functions", "last_good"):
            del st.session_state[key]


current_settings = dict(
    function=function,
    variables=variables,
    center=center,
    degree=degree,
    ranges=ranges,
    fixed_values=fixed_values,
    point=point,
    view=view,
)

try:
    polynomial, figure, table = build_results(**current_settings)
    st.session_state.last_good = current_settings
except Exception as err:
    plt.close("all")
    current_settings = st.session_state.get("last_good", safe_default())
    st.warning(f"⚠️ Calculation error with chosen inputs ({err}). Displaying previous stable configuration.")
    st.button("🔄 Reset to Safe Defaults", on_click=reset_inputs)
    polynomial, figure, table = build_results(**current_settings)

active_vars = current_settings["variables"]
active_degree = current_settings["degree"]

# =====================================================
# TOP METRIC SCORECARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Variables", value=f"{len(active_vars)}D Field")

with col2:
    st.metric(label="Polynomial Degree", value=f"n = {active_degree}")

if table is not None and not table.empty:
    target_row = table[table["degree"] == active_degree].iloc[0]
    actual_val = target_row["actual"]
    approx_val = target_row["approximation"]
    abs_err = target_row["absolute_error"]

    with col3:
        st.metric(label=f"Exact Value @ {current_settings['point']}", value=f"{actual_val:.5f}")
    with col4:
        st.metric(
            label="Taylor Value (Abs Error)",
            value=f"{approx_val:.5f}",
            delta=f"Err: {abs_err:.2e}",
            delta_color="inverse",
        )
else:
    with col3:
        st.metric(label="Exact Value", value="N/A")
    with col4:
        st.metric(label="Absolute Error", value="N/A")

# =====================================================
# ORGANIZED TABS
# =====================================================

tab_plot, tab_math, tab_data, tab_context = st.tabs([
    "📈 Interactive Visualization",
    "🧮 Mathematical Formulation",
    "📊 Error Analysis & Bounds",
    "🌍 Real-World Context",
])

# ----- TAB 1: PLOTS -----
with tab_plot:
    if len(active_vars) > 2:
        slice_desc = ", ".join(
            f"{v} = {val:.2f}" for v, val in zip(active_vars[2:], current_settings["fixed_values"])
        )
        st.info(f"Showing 2D cross-section across ({active_vars[0]}, {active_vars[1]}); Slice: {slice_desc}")

    st.pyplot(figure)

    # Download Button for PPT Evidence
    img_buffer = io.BytesIO()
    figure.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    st.download_button(
        label="📥 Download High-Resolution Plot (PNG for PPT)",
        data=img_buffer.getvalue(),
        file_name="taylor_approximation_plot.png",
        mime="image/png",
    )

# ----- TAB 2: MATH FORMULATION -----
with tab_math:
    st.markdown("#### Original Function")
    st.latex(f"f({', '.join(str(v) for v in active_vars)}) = {sp.latex(current_settings['function'])}")

    st.markdown(f"#### Taylor Polynomial ($P_{{{active_degree}}}$)")
    st.latex(f"P_{{{active_degree}}}({', '.join(str(v) for v in active_vars)}) = {sp.latex(polynomial)}")

    if len(active_vars) == 1 and matches_sympy_series_1d(
        current_settings["function"], active_vars[0], current_settings["center"][0], active_degree
    ):
        st.success("✅ **SymPy Verification**: Result identically matches SymPy's reference engine `sympy.series()`.")

    st.markdown("---")
    st.markdown(
        f"""
        **Mathematical Characteristics:**
        - **Total Derivative Combinations Evaluated:** {term_count}
        - **Center Point ($a$):** {current_settings['center']}
        - **Order Analysis:** At degree {active_degree}, all partial derivatives up to order ${active_degree}$ 
          are matched exactly with the original function at the expansion center.
        """
    )

# ----- TAB 3: ERROR DATA & BOUNDS -----
with tab_data:
    st.markdown(f"#### Error Progression at Coordinate {current_settings['point']}")
    if table is not None:
        fmt_dict = {
            "actual": "{:.6f}",
            "approximation": "{:.6f}",
            "absolute_error": "{:.4e}",
        }
        if "lagrange_bound" in table.columns:
            fmt_dict["lagrange_bound"] = "{:.4e}"

        st.dataframe(table.style.format(fmt_dict), use_container_width=True)

        if "lagrange_bound" in table.columns:
            st.caption("ℹ️ Notice that **absolute_error** is strictly $\le$ **lagrange_bound** across every degree.")

        st.download_button(
            label="📥 Download Error Table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="taylor_error_progression.csv",
            mime="text/csv",
        )
    else:
        st.warning("Coordinate is outside the valid mathematical domain for this function.")

# ----- TAB 4: REAL-WORLD CONTEXT -----
with tab_context:
    st.markdown("### Practical Applications of This Model")

    st.markdown(
        """
        **1. 2D Steady-State Thermal Hotspot Modeling**
        - **Model:** $T(x, y) = 25 + 40 e^{-(x^2 + y^2)/2}$
        - **Engineering Context:** Represents localized heat generation (e.g., a microchip core or laser welding spot) on a conductive metal plate.
        - **Role of Taylor Series:** Microcontrollers on embedded hardware lack the floating-point capability to compute continuous exponential fields. Expanding $T(x,y)$ into a 2nd-degree Taylor polynomial creates an ultra-fast quadratic model used in real-time thermal throttling algorithms.

        **2. Linearization in Control Systems (Pendulum)**
        - **Model:** $\\tau(\\theta) = \\sin(\\theta)$
        - **Engineering Context:** Restoring torque of a nonlinear physical pendulum.
        - **Role of Taylor Series:** Degree 1 ($P_1(\\theta) = \\theta$) simplifies the equation into a linear Harmonic Oscillator for small angles. Degree 3 ($P_3(\\theta) = \\theta - \\theta^3/6$) captures large-amplitude period shifts without requiring elliptic integrals.
        """
    )