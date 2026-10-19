# Methods

## 1. The toy system (Part A)

All variables are monthly; time is in months. Code: `src/knownpred/simulate/system.py`.

### 1.1 ENSO-like recharge oscillator

Following the conceptual recharge oscillator of Jin (1997) in the linear stochastic form of Burgers et al. (2005), with two additions (seasonal damping, multiplicative noise):

$$
dT = \left(-\lambda_T(m)\,T + \omega H\right)dt + \sigma_T\, g(T)\, dW_T, \qquad
dH = \left(-\omega T - \lambda_H H\right)dt + \sigma_H\, dW_H
$$

- $T$: eastern-Pacific SST anomaly (K); $H$: thermocline-depth anomaly in K-equivalent units.
- $\lambda_T(m) = \lambda_{T0} + \lambda_{T1}\cos\left(2\pi (m - 4)/12\right)$: damping is strongest in April, which produces seasonal phase locking (a "spring barrier").
- $g(T) = \max(1 + bT,\ g_{\min})$: state-dependent noise, a simplified form of the noise discussed by Levine & Jin (2010). It produces positive skewness of $T$ and makes forecast spread depend on the ENSO state.
- Integration: Euler–Maruyama with 30 sub-steps per month (Itô interpretation).

### 1.2 Teleconnected precipitation

$$
P_t = \beta(m)\,\bar T_t + s(m, \bar T_t)\,\varepsilon_t, \qquad
\beta(m) = \beta_0 \max\left(0, \cos\left(2\pi (m-1)/12\right)\right)
$$

- $\bar T_t$ is the monthly-mean $T$. The teleconnection is active from November to March only.
- $\varepsilon_t$ is centred Gamma noise with shape $k = 2$ and unit variance, so its skewness is $2/\sqrt{k} \approx 1.4$.
- $s(m, T) = s_0\left[1 + 0.5\cos(2\pi(m-1)/12)\right]\max\left(1 + c\,\tfrac{\beta(m)}{\beta_0}\,T,\ 0.25\right)$. In the cool season, El Niño months are wetter and more variable and La Niña months drier and less variable. This is a second, deliberately built-in source of state-dependent predictability.

### 1.3 Linear-reservoir storage

$$
S_t = a\,S_{t-1} + P_t\cdot\Delta, \qquad a = e^{-1/\tau_S}, \quad \Delta = 365.25/12 \text{ days}
$$

$S$ is a conceptual storage anomaly (mm) with e-folding memory $\tau_S$. It is not a model of any observed storage product.

### 1.4 Parameters

| Parameter | Value | Meaning |
|---|---|---|
| $\omega$ | $2\pi/48$ month$^{-1}$ | ~4-year oscillation |
| $\lambda_{T0}$, $\lambda_{T1}$ | 0.06, 0.05 month$^{-1}$ | annual-mean and seasonal damping of $T$ |
| $\lambda_H$ | 0.02 month$^{-1}$ | damping of $H$ |
| $\sigma_T$, $\sigma_H$ | 0.30, 0.15 | noise amplitudes |
| $b$, $g_{\min}$ | 0.5 K$^{-1}$, 0.2 | multiplicative noise |
| $\beta_0$ | 0.8 mm day$^{-1}$ K$^{-1}$ | teleconnection strength |
| $s_0$, $c$ | 0.4 mm day$^{-1}$, 0.6 K$^{-1}$ | precipitation noise |
| $\tau_S$ | 6 months | storage memory |

The values are illustrative. They give a standard deviation of $T$ of about 0.9 K, positive skewness of about 0.9, a dominant period of about 4.2 years and a variance peak in boreal winter. These properties are checked in `tests/test_system.py`; the parameters are not fitted to observations.

### 1.5 Exact predictability

The full Markov state at the end of month $t$ is $(T, H, S, m)$. All of it is observed by the forecast models, together with two irrelevant AR(1) "distractor" series. The true forecast distribution of $S_{t+\tau}$ is therefore obtained by re-simulating the system from that state many times (1000 continuations per test state; `src/knownpred/truth.py`). The only error is Monte Carlo sampling. The standard error of the true standard deviation is reported, and is about 2–3 % of it for 1000 members.

By construction:

- the true spread depends on $T$, $H$ and the calendar month;
- it does **not** depend on $S$, because the reservoir is linear and $S$ only shifts the mean;
- it does not depend on the distractors.

These are exact statements that attribution methods can be tested against (Q3).

## 2. Forecast models

| Model | Mean | Spread | Code |
|---|---|---|---|
| Climatology | none (issue-month climatology) | issue-month climatology | `baselines.Climatology` |
| Linear | OLS on predictors and their interactions with the annual harmonic | constant within each issue month | `baselines.SeasonalLinear` |
| Gradient boosting (two-stage) | histogram gradient boosting | second booster on log squared out-of-fold residuals, bias-corrected by $E[\log\chi^2_1]$, then one scale factor fitted on validation data | `baselines.TwoStageBoosting` |
| NN (Gaussian) | MLP, 2×64 ReLU | heteroscedastic Gaussian, trained by negative log-likelihood | `models.train.NetForecaster` |
| NN (SHASH) | MLP, 2×64 ReLU | sinh-arcsinh distribution (Jones & Pewsey, 2009): location, scale, skewness, tail weight | same |
| Deep ensemble | 5 Gaussian MLPs with different initialisations | mixture (Lakshminarayanan et al., 2017) | `models.train.DeepEnsemble` |

The networks are trained with Adam, with early stopping on the validation negative log-likelihood (patience 30 epochs) and inputs and target standardised with training statistics only. Seeds are derived per component from the master seed, so adding a model does not change the data or seeds of the other models.

## 3. Data splits

**Part A.** Training, validation and test sets are *independent simulations* with different seeds: 1000, 200 and 500 years. No trajectory is shared between them, so autocorrelation cannot leak information across splits.

**Part B.** CESM2-LE members are split randomly into training, validation and test members (70/15/15 %). Observations are used for evaluation only. All fitted quantities are estimated from training members only: EOF patterns, standardisation and model parameters.

## 4. Evaluation

Outcome-based metrics (available in any application):

- CRPS and CRPS skill score against climatology. The CRPS is computed in closed form for Gaussian forecasts (Gneiting et al., 2005), by quantile integration for SHASH forecasts, and with the "fair" estimator for ensemble forecasts (Ferro, 2014).
- PIT histogram and its deviation statistic D (Nipen & Stull, 2011), with the value expected under perfect calibration for reference.
- Coverage of the central 90 % interval, and the spread–error ratio (RMS predicted σ / RMSE of the mean).
- Discard plots: error of the most-confident fraction of forecasts as the fraction retained shrinks (as in Gordon & Barnes, 2022).

Truth-based metrics (testbed only):

- **Within-month rank correlation** between predicted and true σ. It is computed separately for each issue month and then averaged, which removes the seasonal cycle of spread that any model knowing the calendar can reproduce. A model with constant spread within a month scores 0.
- **Confident-set overlap**: the share of the truly most-predictable 20 % of cases that the model also ranks in its most-confident 20 %, computed within each month. Chance level is 0.2.
- **RMS log spread error**, $\sqrt{\overline{\log^2(\hat\sigma/\sigma)}}$.
- **Mean error against the true mean**, normalised by the mean true σ.
- **Excess CRPS**: CRPS of the model minus CRPS of the ideal forecast on the same outcomes. This paired difference is much less noisy than either score alone.

**Uncertainty of the scores.** 95 % confidence intervals come from a circular block bootstrap (Politis & Romano, 1992). Blocks are 24 months in Part A, longer than the decorrelation time of the target, and 5 years in Part B, where samples are annual. Paired statistics resample both forecasts with the same indices.

**Spread attribution.** Grouped permutation importance of the predicted σ: the mean absolute relative change in σ when one predictor group is shuffled. Calendar month (sine and cosine) is shuffled as one group. The same quantity is computed for the *true* σ by Monte Carlo, using common random numbers so that true zeros are exactly zero. Integrated gradients (Sundararajan et al., 2017) are available for the networks.

## 5. Part B pipeline

1. **Extract.** CESM2-LE `TS` and `PRECC + PRECL` (historical + SSP3-7.0, CMIP6 forcing variant, 1900–2050, all 50 members) are read over HTTPS from the public Zarr stores on AWS. Only the tropical Pacific domain (TS), the U.S. Southwest box (precipitation) and the calendar months the analysis uses are kept. The stores are chunked as one member × 600 months × the full globe, so whole chunks (about 600 of 80–110 MB each) are transferred even though only a small subset is kept. Months are labelled from the midpoint of `time_bnds` (`fix_cesm_time`), which is robust to the end-of-interval time stamps of raw CESM output.
2. **Observations.**
   - ERSSTv5 is regridded bilinearly from its 2° grid to the ~1° CESM grid, with longitudes padded periodically.
   - A common ocean mask is applied to model and observations.
   - nClimGrid precipitation is averaged over the target box with cos(latitude) weights, and monthly totals are converted to mm/day using the actual number of days in each month.
3. **Forced response.**
   - Model: the ensemble mean at each time is removed from each member, leaving internal variability. With 50 members this estimate retains a noise variance of σ²/50.
   - Observations: anomalies from the 1951–2010 climatology, then a quadratic trend is removed from each series. With only one realisation, the forced response has to be estimated statistically; this is a known source of error and is kept simple on purpose.
4. **Predictors** (October conditions): the first six EOFs of tropical Pacific SST anomalies, fitted on training members only, with observations projected onto the model EOFs; the Niño-3.4 index; the July–October Niño-3.4 tendency; and August–October precipitation over the box.
5. **Target.** January–March mean precipitation anomaly over the box, a lead of two months.
6. **Standardisation.** Each dataset is scaled by its own reference statistics: training members for the model, 1951–2010 for observations. This is a simple variance correction for model bias and uses two statistics (mean, standard deviation) from part of the observational record.
7. **Evaluation.** Scores are computed on held-out CESM2-LE members (the perfect-model test) and on observations. For observations the reference is a leave-one-year-out climatology.

## References

- Burgers, G., Jin, F.-F., & van Oldenborgh, G. J. (2005). The simplest ENSO recharge oscillator. *Geophys. Res. Lett.*, 32, L13706.
- Davidian, M., & Carroll, R. J. (1987). Variance function estimation. *J. Amer. Stat. Assoc.*, 82, 1079–1091.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Q. J. R. Meteorol. Soc.*, 140, 1917–1923.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *J. Amer. Stat. Assoc.*, 102, 359–378.
- Gneiting, T., Raftery, A. E., Westveld, A. H., & Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Mon. Wea. Rev.*, 133, 1098–1118.
- Gordon, E. M., & Barnes, E. A. (2022). Incorporating uncertainty into a regression neural network enables identification of decadal state-dependent predictability in CESM2. *Geophys. Res. Lett.*
- Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble forecasts. *Mon. Wea. Rev.*, 129, 550–560.
- Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO. Part I: Conceptual model. *J. Atmos. Sci.*, 54, 811–829.
- Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions. *Biometrika*, 96, 761–780.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 30*.
- Levine, A. F. Z., & Jin, F.-F. (2010). Noise-induced instability in the ENSO recharge oscillator. *J. Atmos. Sci.*, 67, 529–542.
- Nipen, T., & Stull, R. (2011). Calibrating probabilistic forecasts from an NWP ensemble. *Tellus A*, 63, 858–875.
- Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263–270.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proc. ICML*.
