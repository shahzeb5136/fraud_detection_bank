# FraudLens — transaction fraud detection demo

A professional Streamlit demonstration of real-time card fraud detection for a bank, built directly on
`fraud_model_data_dictionary.xlsx`. Every dictionary field is simulated, every *Derived* field is a model
feature, every hygiene check is verified on the data, and every score comes with a complete audit trace.

## Quick start

```bash
pip install -r requirements.txt
python -m fraudlab.build        # ~90 s: synthetic world -> features -> MCMC -> models -> artifacts/
streamlit run app.py
```

(Or simply `streamlit run app.py` and press **Build the demo now** on first launch.)

The approve / monitor / step-up thresholds and the false-decline cost sit under **Decision policy** in the sidebar of every page; Command centre, Live scoring, Customer 360, Portfolio analytics and Model lab all re-evaluate their alerts under it.

`python -m pytest -q` checks that replaying a stored transaction through the live pipeline reproduces its batch features and score, and that stream alerts group into cases.

## What is inside

| Page | What it shows |
|---|---|
| Command centre | Portfolio KPIs, fraud rate over time with Bayesian intervals, alert load, how a transaction is scored |
| Live scoring | Score a hand-built authorisation (7 scenario presets) or replay the unlabelled live window, where alerts are grouped into per-customer cases (a card-testing burst is one case, not twelve alerts); SHAP, tree path, scorecard points, Bayesian posteriors, feature lineage, audit JSON |
| Customer 360 | A cardholder's habits (merchants, hours, amounts, geography), Bayesian propensity and their scored history |
| Portfolio analytics | Fraud on an overall basis: by week, channel, MCC, country, hour; merchant risk league; label maturity; policy load |
| Model lab | Time-based split, out-of-time metrics, ROC/PR/calibration, threshold economics, recall by typology, XGBoost/tree/scorecard/stacker internals, model card |
| Bayesian & MCMC | Model specification, R-hat/ESS, trace plots, posteriors, shrinkage, cold-start predictive, customer propensity |
| Data & dictionary | The dictionary with demo coverage and feature lineage, hygiene ledger, synthetic schema and typologies |
| Audit log | Session decision log with downloadable traces, model registry and artifact hashes |

## Method

* **Synthetic world** (`fraudlab/synth.py`): 1,500 customers with segments, usual merchants, hour profiles,
  subscriptions, travel and devices; 800+ merchants in 24 MCCs; six fraud typologies (card-not-present
  compromise, card testing, account takeover, lost/stolen card-present, fraudulent merchant, friendly fraud);
  disputes with realistic lags, scheme reason codes and resolutions; refunds, reversals, declines, FX.
* **Features** (`fraudlab/features.py`): 78 leakage-safe features; customer history strictly before the
  transaction; merchant risk statistics only from labels matured before scoring time; monthly refits of the
  user-vendor matrix factorisation. The same routine serves batch training and live scoring (parity tested).
* **Models** (`fraudlab/models.py`): decision tree with honest leaf statistics, XGBoost with early stopping,
  isotonic calibration and exact SHAP contributions, logistic scorecard, and a transparent logistic stacker.
* **Bayesian MCMC** (`fraudlab/bayes.py`): hierarchical merchant model (vendor within MCC within population)
  and Beta-Binomial customer propensity, sampled with hand-written NumPy Metropolis-within-Gibbs (4 chains,
  split R-hat, ESS).
* **Scoring** (`fraudlab/scoring.py`): one call produces the probability, the policy tier, expected-loss
  economics and a serialisable trace.

All data is synthetic; the bank is fictional. Fraud is enriched to about 1% of authorisations for the demo.
