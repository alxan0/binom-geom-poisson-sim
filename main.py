import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

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

    # Display empirical statistics
    st.subheader("📊 Statistici empirice vs teoretice")
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

    st.write(
        f"**Medie**: empirică = {empirical_mean:.3f}  |  teoretică = {theo_mean:.3f}\n"
        f"**Varianță**: empirică = {empirical_var:.3f}  |  teoretică = {theo_var:.3f}"
    )

    # Histogram + PMF overlay
    st.subheader("📈 Histogramă vs PMF teoretică")

    fig, ax = plt.subplots()

    # Determine bin edges for histogram (discrete bins)
    k_min, k_max = int(np.min(data)), int(np.max(data))
    bins = np.arange(k_min - 0.5, k_max + 1.5)
    ax.hist(data, bins=bins, density=False, alpha=0.7, rwidth=0.85, label="Date simulate")

    # Overlay theoretical PMF
    k_values = np.arange(k_min, k_max + 1)
    pmf_vals = theoretical_pmf(dist, params, k_values)
    ax.plot(k_values, pmf_vals * len(data), "o-", label="PMF teoretică")

    ax.set_xlabel("k")
    ax.set_ylabel("Frecvență")
    ax.set_title(f"{dist}: Histogramă vs PMF")
    ax.legend()
    st.pyplot(fig)

    # Footer note
    st.info("Distribuția geometrică folosește definiția care numără încercările până la primul succes (k ≥ 1).")
