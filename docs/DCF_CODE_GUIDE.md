# Most Important Code Sections for DCF Results

This guide identifies the critical code sections that directly impact DCF valuation results.

## 🔴 Critical: Core DCF Calculation Logic

### 1. `app/core/dcf.py` - The Heart of DCF Valuation

**`calculate_dcf_value()` (lines 16-82)**
- **Purpose**: Core DCF calculation that converts growth rate paths into enterprise values
- **Key Logic**:
  - Projects cash flows: `fcf_paths[:, year] = fcf_paths[:, year - 1] * (1 + growth_rates[:, year])`
  - Discounts to present value: `pv_projected[:, year] = fcf_paths[:, year] / ((1 + wacc) ** (year + 1))`
  - Calculates terminal value using Gordon Growth Model: `terminal_value = terminal_fcf / (wacc - terminal_growth)`
  - Sums everything: `dcf_values = pv_projected.sum(axis=1) + pv_terminal`
- **Why Critical**: This is where all the financial math happens. Any bug here directly affects all DCF results.

**`dcf_valuation_from_results()` (lines 85-158)**
- **Purpose**: Wrapper that takes simulation results and produces DCF values + analytics
- **Key Logic**:
  - Extracts growth rate paths from simulation results
  - Calls `calculate_dcf_value()` to compute enterprise values
  - Computes analytics (mean, median, percentiles, VaR, CVaR) on DCF values
- **Why Critical**: This is the main entry point for DCF calculations in the app. It connects simulation results to DCF outputs.

**`calculate_equity_value()` (lines 161-187)**
- **Purpose**: Converts enterprise value to equity value per share
- **Key Logic**: `equity_value = (dcf_value + cash - debt) / shares_outstanding`
- **Why Critical**: This is what investors care about - the per-share value.

---

## 🟠 Very Important: Growth Rate Simulation

### 2. `app/core/simulate.py` - Generates Growth Rate Paths

**`simulate_growth_paths()` (lines 72-139)**
- **Purpose**: Generates the growth rate paths that feed into DCF calculations
- **Key Logic**:
  - Samples growth rates for each year from distributions
  - Applies correlation between years if enabled
  - Returns `SimResults` with `paths` array of shape `(n_sims, n_years)`
- **Why Critical**: The quality and distribution of growth rate samples directly determines DCF result distributions. If growth rates are wrong, DCF values are wrong.

**`apply_correlation()` (lines 36-69)**
- **Purpose**: Applies correlation structure between Year 1 and Year 2 growth rates
- **Key Logic**: Uses Cholesky decomposition to transform independent samples into correlated ones
- **Why Critical**: Correlation affects the joint distribution of growth rates, which impacts DCF value distributions (especially tail risks).

**`create_rng()` (lines 17-33)**
- **Purpose**: Creates reproducible random number generator
- **Why Critical**: Ensures results are reproducible. Same seed = same growth rates = same DCF results.

---

## 🟡 Important: Distribution Sampling

### 3. `app/core/distributions.py` - Growth Rate Distributions

**`sample_triangular()` and other distribution samplers**
- **Purpose**: Samples growth rates from probability distributions
- **Why Critical**: The distribution parameters (min, mode, max for triangular) directly determine the range and shape of growth rates, which cascades to DCF results.

---

## 🟢 Important: UI Integration & Display

### 4. `app/streamlit_app.py` - DCF Mode Integration

**DCF Mode Toggle (lines 107-110)**
```python
dcf_mode = st.sidebar.checkbox("DCF Valuation Mode", ...)
```
- **Why Important**: Controls whether DCF calculations are performed

**DCF Parameter Inputs (lines 137-245)**
- Initial FCF input (line ~145)
- WACC input (line ~155)
- Terminal growth/multiple inputs (lines ~165-180)
- Cash, debt, shares inputs for equity calculation (lines ~185-200)
- **Why Important**: These are the user inputs that feed into DCF calculations. Wrong inputs = wrong results.

**Apple Preset Logic (lines 121-135)**
- Sets FCF = $105.0B, WACC = 8.21%, triangular distributions
- **Why Important**: Pre-configures the most important DCF parameters for Apple analysis

**DCF Calculation Call (lines 392-408)**
```python
if dcf_mode_active:
    dcf_values, dcf_analytics = dcf_valuation_from_results(
        results=results,
        initial_fcf=initial_fcf,
        wacc=wacc,
        terminal_growth=terminal_growth,
        terminal_multiple=terminal_multiple,
        fcf_in_billions=True,
    )
```
- **Why Critical**: This is where the UI calls the DCF calculation. This connects user inputs to DCF results.

**DCF Results Display (lines 418-480)**
- Enterprise value metrics (mean, median, percentiles)
- Equity value per share calculation
- Comparison to market cap
- **Why Important**: This is what users see. Must correctly display the DCF results.

**DCF Visualizations (lines 571-632)**
- Distribution histogram of DCF values
- ECDF of DCF values
- **Why Important**: Visual representation of DCF results helps users understand the distribution.

---

## 🔵 Supporting: Analytics & Validation

### 5. `app/core/analytics.py` - Statistical Analysis

**`compute_analytics()`**
- **Purpose**: Computes summary statistics, quantiles, VaR, CVaR on DCF values
- **Why Important**: Provides risk metrics and summary statistics that help interpret DCF results

**`value_at_risk()`, `conditional_var()`, `probability_loss()`**
- **Purpose**: Risk metrics specifically for DCF values
- **Why Important**: These metrics help investors understand downside risk in DCF valuations

### 6. `app/core/validation.py` - Input Validation

**Pydantic models for DCF parameters**
- **Purpose**: Validates that DCF inputs (FCF, WACC, terminal growth) are valid numbers
- **Why Important**: Prevents invalid inputs from causing incorrect DCF calculations

---

## 📊 Data Flow for DCF Results

```
User Inputs (Streamlit UI)
    ↓
1. DCF Parameters (FCF, WACC, terminal growth)
2. Growth Rate Distributions (Year 1, Year 2)
    ↓
simulate_growth_paths() → SimResults with growth rate paths
    ↓
dcf_valuation_from_results() → calls calculate_dcf_value()
    ↓
calculate_dcf_value() → DCF values (enterprise values)
    ↓
calculate_equity_value() → Equity value per share (if cash/debt/shares provided)
    ↓
compute_analytics() → Summary statistics, VaR, CVaR
    ↓
Display in Streamlit UI → Metrics, charts, tables
```

---

## 🎯 Key Takeaways

**Most Critical Files:**
1. **`app/core/dcf.py`** - Contains all DCF calculation logic
2. **`app/core/simulate.py`** - Generates growth rate paths
3. **`app/streamlit_app.py`** (lines 392-480) - Integrates DCF into UI

**Most Critical Functions:**
1. `calculate_dcf_value()` - The actual DCF math
2. `dcf_valuation_from_results()` - Main DCF entry point
3. `simulate_growth_paths()` - Generates inputs for DCF

**Key Parameters That Affect Results:**
1. **Initial FCF** - Starting cash flow (currently $105.0B for Apple)
2. **WACC** - Discount rate (currently 8.21% for Apple)
3. **Growth Rate Distributions** - Year 1 and Year 2 (triangular with min/mode/max)
4. **Terminal Growth** - Perpetual growth rate (currently 3%)
5. **Correlation** - Between Year 1 and Year 2 growth rates

**If you need to modify DCF results, focus on:**
- `calculate_dcf_value()` for calculation logic changes
- Growth rate distributions in `app/core/distributions.py` or UI controls
- DCF parameters in `app/streamlit_app.py` (lines 137-245)

