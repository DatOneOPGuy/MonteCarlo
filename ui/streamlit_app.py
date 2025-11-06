"""
Streamlit UI for Monte Carlo simulations.

Provides an interactive dashboard with controls and visualizations.
"""

from __future__ import annotations

import json
import time
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint (default to localhost)
API_URL = st.sidebar.text_input(
    "API URL", value="http://localhost:8000", help="FastAPI server URL"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_connection() -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def run_simulation(sim_type: str, params: dict) -> dict | None:
    """Run a simulation via API."""
    try:
        if sim_type == "pi":
            response = requests.post(
                f"{API_URL}/simulate/pi",
                json=params,
                timeout=30,
            )
        elif sim_type == "linear":
            response = requests.post(
                f"{API_URL}/simulate/linear",
                json=params,
                timeout=30,
            )
        elif sim_type == "custom":
            response = requests.post(
                f"{API_URL}/simulate/custom",
                json=params,
                timeout=30,
            )
        else:
            st.error(f"Unknown simulation type: {sim_type}")
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


# Header
st.title("🎲 Monte Carlo Simulation Toolkit")
st.markdown(
    "Propagate uncertainty through your models using vectorized Monte Carlo simulations."
)

# Sidebar controls
st.sidebar.header("Simulation Controls")

sim_type = st.sidebar.selectbox(
    "Simulation Type",
    ["π estimate", "Linear model", "Custom model"],
    help="Choose the type of simulation to run",
)

# Common parameters
st.sidebar.subheader("Common Parameters")
n = st.sidebar.number_input(
    "Number of trials (n)",
    min_value=100,
    max_value=10_000_000,
    value=100_000,
    step=10_000,
    help="Number of Monte Carlo trials",
)
if n > 5_000_000:
    st.sidebar.warning("⚠️ Large N may be slow. Consider using a smaller value.")

seed = st.sidebar.number_input(
    "Random seed (optional)",
    min_value=0,
    value=None,
    help="Set seed for reproducibility (leave as None for random)",
)
if seed == 0:
    seed = None

# Type-specific parameters
params = {"n": int(n), "seed": seed if seed is not None else None}

if sim_type == "π estimate":
    st.sidebar.info("Estimates π by sampling points in a unit square and counting those inside a unit circle.")

elif sim_type == "Linear model":
    st.sidebar.subheader("Linear Model Parameters")
    params["mu1"] = st.sidebar.number_input(
        "μ₁ (mean of normal)", value=10.0, step=0.1
    )
    params["sigma1"] = st.sidebar.number_input(
        "σ₁ (std of normal)", min_value=0.01, value=2.0, step=0.1
    )
    params["mu2"] = st.sidebar.number_input(
        "μ₂ (log-space mean of lognormal)", value=1.0, step=0.1
    )
    params["sigma2"] = st.sidebar.number_input(
        "σ₂ (log-space std of lognormal)", min_value=0.01, value=0.25, step=0.01
    )
    params["threshold"] = st.sidebar.number_input(
        "Threshold (optional)", value=None, help="Compute P(y > threshold)"
    )
    if params["threshold"] == 0:
        params["threshold"] = None

elif sim_type == "Custom model":
    st.sidebar.subheader("Distributions")
    st.sidebar.markdown("Add distributions for your custom model.")

    # Distribution builder
    if "dists" not in st.session_state:
        st.session_state.dists = [
            {"name": "x1", "type": "normal", "mu": 0.0, "sigma": 1.0}
        ]

    dist_rows = []
    for i, dist in enumerate(st.session_state.dists):
        with st.sidebar.expander(f"Distribution {i+1}: {dist['name']}", expanded=i==0):
            dist_name = st.text_input("Name", value=dist["name"], key=f"name_{i}")
            dist_type = st.selectbox(
                "Type",
                ["normal", "lognormal", "uniform", "bernoulli", "binomial"],
                index=["normal", "lognormal", "uniform", "bernoulli", "binomial"].index(dist["type"]),
                key=f"type_{i}",
            )
            
            if dist_type == "normal":
                mu = st.number_input("μ", value=dist.get("mu", 0.0), key=f"mu_{i}")
                sigma = st.number_input("σ", min_value=0.01, value=dist.get("sigma", 1.0), key=f"sigma_{i}")
                dist_params = {"mu": mu, "sigma": sigma}
            elif dist_type == "lognormal":
                mu = st.number_input("μ (log-space)", value=dist.get("mu", 0.0), key=f"mu_{i}")
                sigma = st.number_input("σ (log-space)", min_value=0.01, value=dist.get("sigma", 1.0), key=f"sigma_{i}")
                dist_params = {"mu": mu, "sigma": sigma}
            elif dist_type == "uniform":
                a = st.number_input("a (lower)", value=dist.get("a", 0.0), key=f"a_{i}")
                b = st.number_input("b (upper)", value=dist.get("b", 1.0), key=f"b_{i}")
                dist_params = {"a": a, "b": b}
            elif dist_type == "bernoulli":
                p = st.number_input("p", min_value=0.0, max_value=1.0, value=dist.get("p", 0.5), key=f"p_{i}")
                dist_params = {"p": p}
            else:  # binomial
                n_binom = st.number_input("n (trials)", min_value=1, value=int(dist.get("n", 10)), key=f"n_{i}")
                p = st.number_input("p (prob)", min_value=0.0, max_value=1.0, value=dist.get("p", 0.5), key=f"p_binom_{i}")
                dist_params = {"n": float(n_binom), "p": p}

            dist_rows.append({
                "name": dist_name,
                "type": dist_type,
                "params": dist_params,
            })

            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.dists.pop(i)
                st.rerun()

    if st.sidebar.button("➕ Add Distribution"):
        st.session_state.dists.append({"name": f"x{len(st.session_state.dists)+1}", "type": "normal", "mu": 0.0, "sigma": 1.0})
        st.rerun()

    # Model expression
    st.sidebar.subheader("Model Expression")
    model_expr = st.sidebar.text_input(
        "Model (e.g., 'x1 + 2*x2')",
        value="x1 + 2*x2" if len(st.session_state.dists) == 1 else " + ".join([d["name"] for d in st.session_state.dists]),
        help="Mathematical expression using distribution names as variables",
    )
    params["model"] = model_expr
    params["dists"] = dist_rows

    params["threshold"] = st.sidebar.number_input(
        "Threshold (optional)", value=None, help="Compute P(y > threshold)"
    )
    if params["threshold"] == 0:
        params["threshold"] = None

# Run button
if st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True):
    if not check_api_connection():
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running:\n```bash\nuvicorn montecarlo.api:app --reload\n```")
    else:
        with st.spinner("Running simulation..."):
            start_time = time.time()
            # Map UI selection to API endpoint name (case-insensitive)
            api_type_map = {
                "π estimate": "pi",
                "linear model": "linear",
                "custom model": "custom"
            }
            # Normalize the sim_type for matching (case-insensitive)
            normalized_type = sim_type.strip()
            # Try exact match first
            api_type = api_type_map.get(normalized_type)
            # If no exact match, try case-insensitive match
            if api_type is None:
                normalized_lower = normalized_type.lower()
                for key, value in api_type_map.items():
                    if key.lower() == normalized_lower:
                        api_type = value
                        break
            # If still no match, use fallback transformation
            if api_type is None:
                api_type = normalized_type.lower().replace(" ", "_").replace("π", "pi")
                # Clean up common patterns
                api_type = api_type.replace("_model", "").replace("_estimate", "")
            result = run_simulation(api_type, params)
            elapsed_ms = (time.time() - start_time) * 1000

        if result:
            st.session_state["last_result"] = result
            st.session_state["last_params"] = params
            st.session_state["last_sim_type"] = sim_type  # Store the sim_type used
            st.session_state["elapsed_ms"] = elapsed_ms
            st.success("✅ Simulation completed!")

# Display results
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    params = st.session_state["last_params"]
    stored_sim_type = st.session_state.get("last_sim_type", sim_type)  # Use stored sim_type
    elapsed_ms = st.session_state.get("elapsed_ms", 0)

    # Metrics cards
    st.header("Results")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean", f"{result.get('mean', result.get('pi_estimate', 0)):.4f}")
    with col2:
        if "std" in result:
            st.metric("Std Dev", f"{result['std']:.4f}")
        else:
            st.metric("Std Error", f"{result.get('se', 0):.6f}")
    with col3:
        if "p05" in result:
            st.metric("5th %ile", f"{result['p05']:.4f}")
        else:
            st.metric("Trials", f"{result['n']:,}")
    with col4:
        if "p50" in result:
            st.metric("Median", f"{result['p50']:.4f}")
        else:
            st.metric("Seed", str(result.get("seed", "None")))
    with col5:
        if "p95" in result:
            st.metric("95th %ile", f"{result['p95']:.4f}")
        else:
            st.metric("Runtime", f"{elapsed_ms:.1f} ms")

    # Extra metrics
    if "extras" in result and result["extras"]:
        extras = result["extras"]
        if "P(y > threshold)" in extras:
            st.metric(
                "P(y > threshold)",
                f"{extras['P(y > threshold)']:.4f}",
                delta=f"±{extras.get('P_se', 0):.4f}",
            )

    # Tabs for visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Convergence", "📋 Details"])

    with tab1:
        # For visualization, we need to run the simulation again with keep=True
        # Since API doesn't return all samples, we'll simulate locally for visualization
        st.subheader("Distribution Histogram with KDE")
        
        # Use stored params and sim_type from the actual simulation
        viz_params = params  # Use stored params
        viz_sim_type = stored_sim_type  # Use stored sim_type
        viz_n = min(viz_params.get("n", n), 100_000)  # Limit for visualization
        viz_seed = viz_params.get("seed", seed)
        
        try:
            if viz_sim_type == "π estimate":
                def model(x):
                    inside = (x[:, 0]**2 + x[:, 1]**2) <= 1.0
                    return inside.astype(float) * 4  # Scale to π estimate
                
                def sampler(rng, size):
                    return rng.uniform(0, 1, size=(size, 2))
                
                from montecarlo.core import Simulation
                sim = Simulation(model, sampler, seed=viz_seed)
                viz_result = sim.run(viz_n, keep=True)
                y = viz_result["all"]
                
            elif viz_sim_type == "Linear model":
                def model(x):
                    return x[:, 0] + 2 * x[:, 1]
                
                def sampler(rng, size):
                    from montecarlo.distributions import normal, lognormal
                    x1 = normal(viz_params["mu1"], viz_params["sigma1"])(rng, size)
                    x2 = lognormal(viz_params["mu2"], viz_params["sigma2"])(rng, size)
                    return np.column_stack([x1, x2])
                
                from montecarlo.core import Simulation
                sim = Simulation(model, sampler, seed=viz_seed)
                viz_result = sim.run(viz_n, keep=True)
                y = viz_result["all"]
                
            else:  # custom
                # Build custom model locally
                from montecarlo.core import Simulation
                from montecarlo.distributions import normal, lognormal, uniform, bernoulli, binomial
                
                dist_factories = {
                    "normal": normal,
                    "lognormal": lognormal,
                    "uniform": uniform,
                    "bernoulli": bernoulli,
                    "binomial": binomial,
                }
                
                samplers = []
                var_names = []
                for dist_spec in viz_params["dists"]:
                    factory = dist_factories[dist_spec["type"]]
                    if dist_spec["type"] == "normal":
                        sampler_fn = factory(dist_spec["params"]["mu"], dist_spec["params"]["sigma"])
                    elif dist_spec["type"] == "lognormal":
                        sampler_fn = factory(dist_spec["params"]["mu"], dist_spec["params"]["sigma"])
                    elif dist_spec["type"] == "uniform":
                        sampler_fn = factory(dist_spec["params"]["a"], dist_spec["params"]["b"])
                    elif dist_spec["type"] == "bernoulli":
                        sampler_fn = factory(dist_spec["params"]["p"])
                    else:  # binomial
                        sampler_fn = factory(int(dist_spec["params"]["n"]), dist_spec["params"]["p"])
                    samplers.append(sampler_fn)
                    var_names.append(dist_spec["name"])
                
                def sampler(rng, size):
                    samples = [s(rng, size) for s in samplers]
                    return np.column_stack(samples)
                
                def model(x):
                    variables = {name: x[:, i] for i, name in enumerate(var_names)}
                    # Safe eval with limited context
                    safe_dict = {
                        "__builtins__": {},
                        "np": np,
                        "abs": abs,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "sqrt": np.sqrt,
                        "exp": np.exp,
                        "log": np.log,
                        "sin": np.sin,
                        "cos": np.cos,
                        "tan": np.tan,
                        "pi": np.pi,
                        "e": np.e,
                    }
                    safe_dict.update(variables)
                    try:
                        return eval(viz_params["model"], {"__builtins__": {}}, safe_dict)
                    except Exception as e:
                        st.error(f"Model evaluation error: {e}")
                        return np.zeros(len(x))
                
                sim = Simulation(model, sampler, seed=viz_seed)
                viz_result = sim.run(viz_n, keep=True)
                y = viz_result["all"]
            
            from montecarlo.viz import hist_with_kde
            fig = hist_with_kde(y, bins=60, title="Output Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {e}")
            st.info("Run the simulation first to see visualizations.")

    with tab2:
        st.subheader("Convergence Plot")
        try:
            # Generate convergence data (subsample for performance)
            if "y" in locals():
                step = max(1, len(y) // 10_000)  # Subsample to 10k points
                y_subsample = y[::step]
                
                from montecarlo.viz import convergence_plot
                true_val = None
                if stored_sim_type == "π estimate":
                    true_val = np.pi
                fig = convergence_plot(y_subsample, true_value=true_val, title="Cumulative Mean Convergence")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run the simulation first to see convergence plot.")
        except Exception as e:
            st.error(f"Convergence plot error: {e}")

    with tab3:
        st.subheader("Simulation Details")
        
        # JSON summary
        st.markdown("**Summary Statistics:**")
        st.json(result)
        
        st.markdown("**Parameters Used:**")
        st.json(params)
        
        # CSV download (if samples available)
        if "y" in locals() and len(y) <= 100_000:
            st.markdown("**Download Samples:**")
            df = pd.DataFrame({"value": y})
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="monte_carlo_samples.csv",
                mime="text/csv",
            )
        else:
            st.info("Sample download available for simulations with ≤100k trials. Run a smaller simulation to download.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Monte Carlo Toolkit**  
    Version 0.1.0
    """
)

if __name__ == "__main__":
    pass

