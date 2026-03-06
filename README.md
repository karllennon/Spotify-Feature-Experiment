# Spotify Feature Launch Experiment Pipeline

A three-stage A/B testing framework simulating how a data team would measure the impact of a new AI-powered playlist feature on free-to-paid Premium conversion. Built to demonstrate statistical experimentation skills including hypothesis testing, confidence intervals, Bayesian inference, and multiple testing correction.

## Business Question

Spotify launches a new AI-powered playlist feature. Does it drive free-to-paid conversion among free-tier users?

## Experiment Design

10,000 free-tier users are randomly assigned to control and treatment groups with equal 50/50 probability. Each user passes through a three-stage funnel modeled as independent Bernoulli trials with group-specific rates.

| Stage | Description |
|-------|-------------|
| Feature Adoption | Did the user engage with the new AI playlist feature? |
| Notification Re-engagement | For non-adopters, did a push notification bring them back? |
| Premium Conversion | For adopters, did a targeted upgrade prompt convert them to paid? |

## Statistical Methods

- Chi-square test of independence (all three stages)
- Two-sample independent t-test (conversion stage)
- Wilson confidence intervals on observed lift
- Bayesian Beta-Binomial comparison via Monte Carlo sampling
- Bonferroni correction for multiple testing

## Results

| Stage | Control | Treatment | Lift | Significant |
|-------|---------|-----------|------|-------------|
| Feature Adoption | 5.3% | 25.7% | +20.4pp | Yes |
| Notification Re-engagement | 7.5% | 14.3% | +6.8pp | Yes |
| Premium Conversion | 0.4% | 4.3% | +4.0pp | Yes |

All results survive Bonferroni correction. Bayesian analysis assigns 100% posterior probability to treatment outperforming control at every stage.

## Project Structure
```
spotify-experiment/
├── data/                        # Simulated dataset output
├── notebooks/
│   └── experiment_analysis.ipynb  # Full narrative analysis
├── src/
│   ├── simulate.py              # Three-stage funnel data simulation engine
│   ├── stats_engine.py          # Statistical analysis engine
│   └── utils.py                 # Helper functions
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
└── README.md
```

## Tech Stack

Python, NumPy, Pandas, SciPy, Statsmodels, Plotly, Streamlit

## Setup
```bash
conda create -n spotify-experiment python=3.11
conda activate spotify-experiment
pip install numpy pandas scipy statsmodels plotly streamlit jupyter
```

## Running the Project

Run the simulation and analysis:
```bash
python src/simulate.py
python src/stats_engine.py
```

Launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/experiment_analysis.ipynb
```