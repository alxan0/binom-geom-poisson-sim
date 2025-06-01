import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom, geom, poisson, chisquare
# ------------------------------
# Helper functions
# ------------------------------

def simulate_data(dist: str, params: dict, size: int) -> np.ndarray:
    """Generate random samples for the chosen distribution."""
    if dist == "Binomial":
        n, p = params["n"], params["p"]
        return np.random.binomial(n=n, p=p, size=size)
    elif dist == "Geometric":
        # In numpy, the geometric distribution counts the number of trials up to and including the first success.
        p = params["p"]
        return np.random.geometric(p=p, size=size)
    elif dist == "Poisson":
        lam = params["lam"]
        return np.random.poisson(lam=lam, size=size)
    else:
        raise ValueError("Unsupported distribution")


def theoretical_pmf(dist: str, params: dict, k_values: np.ndarray) -> np.ndarray:
    """Return theoretical PMF for overlay on histogram."""
    if dist == "Binomial":
        n, p = params["n"], params["p"]
        return binom.pmf(k_values, n=n, p=p)
    elif dist == "Geometric":
        p = params["p"]
        # numpy/geometric counts trials until first success, so support is k=1,2,...
        return (1 - p) ** (k_values - 1) * p
    elif dist == "Poisson":
        lam = params["lam"]
        return poisson.pmf(k_values, mu=lam)
    else:
        raise ValueError("Unsupported distribution")


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Simulare distribuții discrete", page_icon="🎲")
st.title("🎲 Simularea distribuțiilor Binomială, Geometrică și Poisson")

st.markdown(
    """
    Selectează distribuția, setează parametrii și apasă **Simulează** pentru a genera valori.
    Histograma rezultată este comparată cu funcția de masă a probabilității (PMF) teoretică.
    """
)

# Sidebar controls
with st.sidebar:
    dist = st.selectbox("Alege distribuția", ("Binomial", "Geometric", "Poisson"))

    if dist == "Binomial":
        n = st.number_input("n (număr de încercări)", min_value=1, value=10, step=1)
        p = st.slider("p (probabilitate de succes)", min_value=0.0, max_value=1.0, value=0.5)
        params = {"n": int(n), "p": p}
    elif dist == "Geometric":
        p = st.slider("p (probabilitate de succes)", min_value=0.0, max_value=1.0, value=0.25)
        params = {"p": p}
    else:  # Poisson
        lam = st.number_input("λ (rată medie)", min_value=0.0, value=5.0)
        params = {"lam": lam}

    size = st.number_input("Număr de observații", min_value=100, value=1000, step=100)
    simulate_btn = st.button("Simulează")

if simulate_btn:
    # Generate data
    data = simulate_data(dist, params, int(size))

    # ---------------------------------------------------------------------
    # 1) Histogram + PMF overlay
    # ---------------------------------------------------------------------
  
    st.subheader("📈 Histogramă vs PMF teoretică")

    # Histogram (empirical)
    k_min, k_max = int(data.min()), int(data.max())
    bins = np.arange(k_min - 0.5, k_max + 1.5)

    # Theoretical PMF - expected frequencies
    k_vals   = np.arange(k_min, k_max + 1)
    pmf_vals = theoretical_pmf(dist, params, k_vals) 
    expected = pmf_vals * len(data)

    # Clean Histogram + PMF overlay
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, alpha=0.7, rwidth=0.85, label="Date simulate")
    ax.plot(k_vals, expected, "o-", label="PMF teoretică", color="C1")

    ax.set_xlabel("k")
    ax.set_ylabel("Frecvență")
    ax.set_title(f"{dist}: Histogramă vs PMF")
    ax.legend()
    st.pyplot(fig)

    # Detailed Histogram + PMF overlay
    fig, ax = plt.subplots()
    counts, _, _ = ax.hist(data, bins=bins, alpha=0.7, rwidth=0.85, label="Date simulate")
    ax.plot(k_vals, expected, "o-", label="PMF teoretică", color="C1")

    # Labels: empirical above the bar, theoretical to the right of the point
    offset_x = 0.6  # small right shift
    offset_y = -1   # small downward shift
    for idx, k in enumerate(k_vals):
        # empirical count label
        ax.text(k, counts[idx] + offset_y, f"{int(counts[idx])}", ha="center", va="top", fontsize=8)
        # theoretical expected count label
        ax.text(k + offset_x, expected[idx], f"{expected[idx]:.1f}", ha="center", va="bottom", fontsize=8, color="C1")

    # If labels stick out on the right, add a bit more margin
    fig.subplots_adjust(right=0.95)

    ax.set_xlabel("k")
    ax.set_ylabel("Frecvență")
    ax.set_title(f"{dist}: Histogramă vs PMF")
    ax.legend()
    st.pyplot(fig)

    # ---------------------------------------------------------------------
    # 2) Table with probabilities & differences
    # ---------------------------------------------------------------------
    
    st.subheader("📋 Statistici empirice vs teoretice")

    empirical_prob   = counts / len(data)
    theoretical_prob = pmf_vals
    diff             = empirical_prob - theoretical_prob

    table = pd.DataFrame({
        "k": k_vals,
        "Empirical P(k)": empirical_prob.round(4),
        "Theoretical P(k)": theoretical_prob.round(4),
        "Difference": diff.round(4),
    })
    st.dataframe(table, hide_index=True)

    # E(x) and Var(x)
    empirical_mean = np.mean(data)
    empirical_var = np.var(data, ddof=1)

    if dist == "Binomial":
        theo_mean, theo_var = params["n"] * params["p"], params["n"] * params["p"] * (1 - params["p"])
    elif dist == "Geometric":
        p = params["p"]
        theo_mean, theo_var = 1 / p, (1 - p) / (p ** 2)
    else:  # Poisson
        lam = params["lam"]
        theo_mean, theo_var = lam, lam

    st.markdown(
        f"""
        **Medie**: empirică = {empirical_mean:.3f} | teoretică = {theo_mean:.3f}  
        **Varianță**: empirică = {empirical_var:.3f} | teoretică = {theo_var:.3f}
        """
    )

    # χ^2 goodness‑of‑fit test (using counts & scaled expected counts)
    expected *= len(data) / expected.sum()
    chi_stat, p_val = chisquare(f_obs=counts, f_exp=expected)
    alpha = 0.05
    verdict = "✅ Eșantionul se potrivește cu modelul (p ≥ 0.05)" if p_val >= alpha else "❌ Respingem modelul (p < 0.05)"

    # Largest absolute discrepancy
    max_idx = int(np.argmax(np.abs(diff)))
    max_k   = int(k_vals[max_idx])
    max_gap = diff[max_idx]

    st.subheader("📝 Concluzii")
    st.markdown(
        f"""
        *Statistica χ²* : **{chi_stat:.2f}**,   *p-valoare*: **{p_val:.3f}**  →  {verdict}  
        Cea mai mare diferență absolută de probabilitate la **k = {max_k}**: {max_gap:+.4f}
        """
    )