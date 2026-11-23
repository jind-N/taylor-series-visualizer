import math

import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from errors import compare_at_point
from plots import plot_1d, plot_2d, plot_surface
from taylor import taylor_polynomial


st.set_page_config(page_title="Taylor Series Visualizer", layout="wide")
st.title("Interactive Taylor Series Visualizer")
st.caption("Choose a function, a center and a degree. Everything else is automatic.")

# ---------- Predefined functions: label -> formula ----------
PREDEFINED = {
    "sin(x)": "sin(x)",
    "cos(x)": "cos(x)",
    "e^x": "exp(x)",
    "ln(1 + x)": "log(1 + x)",
    "1 / (1 - x)": "1/(1 - x)",
    "e^(-x²)  (bell curve)": "exp(-x**2)",
    "arctan(x)": "atan(x)",
    "e^(x + y)": "exp(x + y)",
    "sin(x)·cos(y)": "sin(x)*cos(y)",
    "e^(-(x² + y²))  (dome)": "exp(-(x**2 + y**2))",
    "e^x·sin(y)": "exp(x)*sin(y)",
    "e^(x + y + z)": "exp(x + y + z)",
    "sin(x)·cos(y)·e^z": "sin(x)*cos(y)*exp(z)",
    "Real-world: thermal hotspot T(x, y)": "25 + 40*exp(-(x**2 + y**2)/2)",
    "Real-world: pendulum torque sin(x)": "sin(x)",
}

DEGREE_LABELS = {
    0: "0 – Constant (flat line)",
    1: "1 – Linear (straight line)",
    2: "2 – Quadratic (one bend)",
    3: "3 – Cubic (two bends)",
}

MAX_TERMS = 120  # safety limit so the app never freezes

# Functions the user saves during this session
if "saved_functions" not in st.session_state:
    st.session_state.saved_functions = {}


def read_function(text):
    """Turn typed text into a SymPy expression. Accepts ^ and 2x style input."""
    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )
    return parse_expr(text, transformations=transformations)


# =====================================================
# SIDEBAR
# =====================================================

# ---------- 1. Function ----------
st.sidebar.header("1. Function")

mode = st.sidebar.radio(
    "How do you want to choose?",
    ["Pick from list", "Type my own"],
    horizontal=True,
    help="Pick a ready-made function, or type any formula.",
)

if mode == "Pick from list":
    all_functions = {**PREDEFINED}
    for label, formula in st.session_state.saved_functions.items():
        all_functions[f"⭐ {label}"] = formula

    choice = st.sidebar.selectbox(
        "Search or select a function",
        list(all_functions.keys()),
        help="Click and start typing to search, e.g. 'sin' or 'real'.",
    )
    function_text = all_functions[choice]

else:
    function_text = st.sidebar.text_input(
        "Type any function",
        value="x^2 + sin(y)",
        help="Examples: x^2*y, exp(-x^2), log(1+x), sin(x)*cos(y)*z",
    )
    if st.sidebar.button("Save to my list",
                         help="Adds this function to the searchable list."):
        st.session_state.saved_functions[function_text] = function_text
        st.sidebar.success("Saved. Find it under 'Pick from list'.")

try:
    function = read_function(function_text)
except Exception:
    st.error("Could not understand the function. Check brackets and spelling.")
    st.stop()

variables = tuple(sorted(function.free_symbols, key=lambda s: s.name))
if not variables:
    st.error("The function needs at least one variable, e.g. x.")
    st.stop()

# ---------- 2. Center ----------
st.sidebar.header("2. Center")

center = tuple(
    st.sidebar.number_input(
        f"Center for {v}",
        value=0.0,
        step=0.5,
        format="%.2f",
        key=f"center_{v}",
        help="Type any value. The approximation is most accurate here.",
    )
    for v in variables
)

# ---------- 3. Degree ----------
st.sidebar.header("3. Degree")

degree = st.sidebar.select_slider(
    "Highest power used",
    options=list(range(0, 9)),
    value=2,
    format_func=lambda n: DEGREE_LABELS.get(n, f"{n} – Higher accuracy"),
    help="Higher degree = more terms = closer match to the function.",
)

term_count = math.comb(degree + len(variables), len(variables))
if term_count > MAX_TERMS:
    st.warning(
        f"This needs {term_count} derivatives, which is too slow. "
        "Lower the degree or use fewer variables."
    )
    st.stop()

# ---------- 4. View (only for 2+ variables) ----------
view = "3D surface"
if len(variables) > 1:
    st.sidebar.header("4. View")
    view = st.sidebar.radio(
        "Graph style", ["3D surface", "Heatmap"], horizontal=True,
        help="Surface: height = value. Heatmap: colour = value.",
    )

# ---------- Automatic settings (can be overridden) ----------
view_width = 3.0
point = tuple(c + 0.5 for c in center)
fixed_values = list(center[2:])

with st.sidebar.expander("Advanced (optional)"):
    st.caption("Set automatically. Change only if needed.")

    view_width = st.number_input(
        "View distance around the center",
        min_value=0.1,
        value=3.0,
        step=0.5,
        help="The graph shows center − distance to center + distance.",
    )

    if st.checkbox("Choose my own test point",
                   help="Default: 0.5 away from the center."):
        point = tuple(
            st.number_input(f"Test value of {v}", value=float(p), step=0.1,
                            format="%.2f", key=f"point_{v}")
            for v, p in zip(variables, point)
        )

    if len(variables) > 2 and st.checkbox(
        "Choose slice values",
        help="Default: extra variables are fixed at their center.",
    ):
        fixed_values = [
            st.number_input(f"Fixed value of {v}", value=float(c), step=0.1,
                            format="%.2f", key=f"fixed_{v}")
            for v, c in zip(variables[2:], center[2:])
        ]

ranges = tuple((c - view_width, c + view_width) for c in center)

# =====================================================
# MAIN PAGE: Function -> Polynomial -> Graph -> Table
# =====================================================

# =====================================================
# SAFETY NET: fall back to the last working settings
# =====================================================

def build_results(function, variables, center, degree,
                  ranges, fixed_values, point, view):
    """Compute the polynomial, the figure and the table. May raise errors."""
    polynomial = taylor_polynomial(function, variables, center, degree)

    if len(variables) == 1:
        polynomials = {
            d: taylor_polynomial(function, variables, center, d)
            for d in sorted({1, degree})
        }
        low, high = ranges[0]
        figure = plot_1d(function, polynomials, variables[0], center[0], low, high)
    else:
        plot_function = plot_surface if view == "3D surface" else plot_2d
        figure = plot_function(
            function, polynomial, variables, degree,
            ranges=(ranges[0], ranges[1]), fixed_values=fixed_values,
        )

    # The table is optional: if the test point is invalid, skip only the table
    try:
        table = compare_at_point(function, variables, center, point, max_degree=degree)
    except Exception:
        table = None

    return polynomial, figure, table


def safe_default_settings():
    """Settings that are guaranteed to work."""
    x = sp.symbols("x")
    return dict(
        function=sp.sin(x), variables=(x,), center=(0.0,), degree=2,
        ranges=((-3.0, 3.0),), fixed_values=[], point=(0.5,), view="3D surface",
    )


def reset_inputs():
    """Clear typed centers/test points, keeping saved functions."""
    for key in list(st.session_state.keys()):
        if key not in ("saved_functions", "last_good"):
            del st.session_state[key]


settings = dict(
    function=function, variables=variables, center=center, degree=degree,
    ranges=ranges, fixed_values=fixed_values, point=point, view=view,
)

try:
    polynomial, figure, table = build_results(**settings)
    st.session_state.last_good = settings          # remember what worked
except Exception as error:
    plt.close("all")
    settings = st.session_state.get("last_good", safe_default_settings())
    st.warning(
        f"These settings don't work ({error}). "
        "Showing the last settings that worked instead."
    )
    st.button("Reset to safe values", on_click=reset_inputs)
    polynomial, figure, table = build_results(**settings)

# =====================================================
# MAIN PAGE: Function -> Polynomial -> Graph -> Table
# =====================================================

shown_vars = settings["variables"]

st.subheader("Function")
st.latex(f"f({', '.join(str(v) for v in shown_vars)}) = "
         f"{sp.latex(settings['function'])}")
st.caption(f"Detected variables: {', '.join(str(v) for v in shown_vars)} "
           f"({len(shown_vars)}-variable function)")

st.subheader(f"Taylor polynomial of degree {settings['degree']} "
             f"around {settings['center']}")
st.latex(f"P_{{{settings['degree']}}} = {sp.latex(polynomial)}")

st.subheader("Graph")
if len(shown_vars) > 2:
    st.info(
        f"Showing {shown_vars[0]} and {shown_vars[1]}; "
        + ", ".join(f"{v} = {val}" for v, val in
                    zip(shown_vars[2:], settings["fixed_values"]))
        + " (held constant)."
    )
st.pyplot(figure)
plt.close(figure)

st.subheader("Accuracy check")
if table is not None:
    st.caption(f"Real value vs approximation at the test point {settings['point']}")
    st.dataframe(table, use_container_width=True)
else:
    st.info("The function is undefined at the test point, so the table is skipped. "
            "Change the test point in Advanced.")