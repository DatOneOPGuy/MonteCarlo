# Apple DCF Valuation Guide

This guide explains how to use the Monte Carlo DCF simulator for your Apple valuation project.

> **📚 For a detailed explanation of why Monte Carlo simulation and FCF growth rates are used, see [METHODOLOGY_EXPLANATION.md](METHODOLOGY_EXPLANATION.md)**

## Quick Start for Apple Analysis

### Step 1: Enable DCF Mode
1. Open the app: `streamlit run app/streamlit_app.py`
2. In the sidebar, check **"DCF Valuation Mode"**

### Step 2: Set DCF Parameters

**Initial Free Cash Flow (Year 0)**
- Use Apple's most recent FCF from their 10-K or financial statements
- Apple's FY2025 FCF: **$105.0 billion** (already set as default in preset)

**WACC (Discount Rate)**
- Your project uses **7.66%** (0.0766)
- This is already set as the default

**Terminal Value Method**
- Choose **"Perpetual Growth"** for Gordon Growth Model
- Set terminal growth rate (e.g., 3% = 0.03)
- Or use **"Terminal Multiple"** if you prefer that approach

### Step 3: Configure Growth Rate Distributions

**Year 1 Growth Rate**
- Select distribution type (Normal, Lognormal, Student-t, etc.)
- Set parameters based on your uncertainty assumptions
- Example for Normal: μ = 0.05 (5% mean growth), σ = 0.02 (2% volatility)

**Year 2 Growth Rate**
- Configure similarly to Year 1
- Consider correlation between years (enable correlation matrix if needed)

### Step 4: Set Simulation Parameters

- **Number of simulations**: 10,000+ recommended for stable results
- **Random seed**: Use a fixed seed (e.g., 42) for reproducibility
- **LHS**: Optional, provides better coverage with fewer samples

### Step 5: Run and Analyze

1. Click **"Run Simulation"**
2. View results:
   - **DCF Valuation Results**: Mean, median, percentiles of enterprise value
   - **Equity Value per Share** (if you enable it and enter cash/debt/shares)
   - **Distribution charts**: See the range of possible valuations
   - **ECDF**: Cumulative probability of different valuation levels

### Step 6: Compare to Market Price

- Compare the **median DCF value** to Apple's current market capitalization
- Use the **5th and 95th percentiles** to show valuation range
- Export CSV for further analysis

## Example Configuration for Apple

Based on your project parameters:

```
DCF Mode: Enabled
Initial FCF: [Apple's FY2024 FCF - check 10-K]
WACC: 0.0766 (7.66%)
Terminal Growth: 0.03 (3%)

Year 1:
  Distribution: Normal
  μ: 0.05 (5% mean growth)
  σ: 0.02 (2% volatility)

Year 2:
  Distribution: Normal
  μ: 0.06 (6% mean growth)
  σ: 0.03 (3% volatility)

Correlation: [Optional - set if Year 1 and Year 2 growth are correlated]

Simulations: 10,000
Seed: 42 (for reproducibility)
```

## Interpreting Results

### Enterprise Value (DCF Value)
- This is the total value of Apple's operations
- Compare to market cap to assess if stock is over/under-valued

### Equity Value per Share
- If you enable this and enter:
  - Cash & equivalents
  - Total debt
  - Shares outstanding
- You'll get per-share valuation to compare directly to stock price

### Percentiles
- **5th percentile**: Conservative valuation (only 5% of scenarios lower)
- **Median (50th)**: Expected valuation
- **95th percentile**: Optimistic valuation (only 5% of scenarios higher)

## Exporting Results

1. **CSV Export**: Contains all simulation paths and DCF values
2. **JSON Export**: Configuration parameters for reproducibility
3. Use the parameter hash in the footer to track specific runs

## Tips for Your Project

1. **Reproducibility**: Always use the same seed when comparing scenarios
2. **Sensitivity Analysis**: Try different growth rate assumptions to see impact
3. **Correlation**: If Year 1 and Year 2 growth are correlated (e.g., if AI catch-up happens in Year 1, it affects Year 2), enable correlation
4. **Terminal Growth**: Conservative terminal growth (2-3%) is typical for mature companies
5. **Multiple Scenarios**: Run Base/Bull/Bear scenarios with different growth assumptions

## Comparing to Current Stock Price

After running the simulation:
1. Note the median DCF value (enterprise value)
2. Add cash, subtract debt to get equity value
3. Divide by shares outstanding to get per-share value
4. Compare to current stock price
5. Use the 5th-95th percentile range to show uncertainty

## Questions to Address in Your Analysis

1. **Is Apple overvalued or undervalued?** Compare median DCF to market cap
2. **What's the range of possible valuations?** Use 5th-95th percentiles
3. **How sensitive is valuation to growth assumptions?** Try different distributions
4. **What's the probability of significant over/undervaluation?** Use ECDF chart
5. **How does correlation affect results?** Compare correlated vs. independent scenarios

Good luck with your project!

