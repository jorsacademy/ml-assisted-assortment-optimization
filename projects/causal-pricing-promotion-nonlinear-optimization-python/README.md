# Causal Pricing & Promotion Nonlinear Optimization

A compact decision-science example that combines:

1. synthetic observational pricing data,
2. a nonlinear S-Learner using LightGBM,
3. counterfactual dose-response estimation,
4. saturating response-curve fitting,
5. constrained nonlinear budget allocation with SciPy SLSQP.

The implementation is intentionally independent rather than a transcription of the reference article.

## Why this version is different

- uses a train/test split instead of reporting training performance only;
- estimates the treatment response by averaging counterfactual predictions over many real covariate profiles rather than fixing every confounder to its mean;
- fits response curves with positive parameter bounds;
- validates budget feasibility before optimization;
- checks optimizer convergence and constraint residuals;
- separates simulation, causal-response estimation, curve fitting and optimization into reusable modules.

## Run

```bash
python -m pip install -e .
python main.py
```

The script prints predictive metrics, fitted response-curve parameters and the optimal promotional allocation.

## Method

For product \(j\), an S-Learner estimates:

\[
\hat m_j(x,t) \approx E[Y \mid X=x,T=t].
\]

A population-level incremental response curve is approximated by:

\[
\hat\tau_j(t)
=
\frac{1}{n}\sum_i
\left[
\hat m_j(X_i,t)-\hat m_j(X_i,0)
\right].
\]

A saturating Hill curve is then fitted:

\[
g_j(t)=A_j \frac{t^{h_j}}{K_j^{h_j}+t^{h_j}}.
\]

Finally:

\[
\max_{t_1,\ldots,t_J}
\sum_j g_j(t_j)
\]

subject to a shared budget and product-level bounds.

## Caveat

An S-Learner is not automatically causal. Identification still requires assumptions such as no unmeasured confounding, consistency and adequate treatment overlap. Synthetic data are used here so the full pipeline can be demonstrated cleanly.
