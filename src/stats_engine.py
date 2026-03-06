import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.proportion as smp

# Constructs a 2x2 contingency table for a given binary outcome column.
# Returns a 2D array structured as [[control_yes, control_no], [treatment_yes, treatment_no]].
def build_contingency_table(df: pd.DataFrame, outcome_col: str) -> np.ndarray:
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    control_yes = control[outcome_col].sum()
    control_no = len(control) - control_yes
    treatment_yes = treatment[outcome_col].sum()
    treatment_no = len(treatment) - treatment_yes

    return np.array([
        [control_yes, control_no],
        [treatment_yes, treatment_no]
    ])


# Runs a chi-square test of independence on a 2x2 contingency table.
# Returns the chi-square statistic, p-value, and whether the result is significant.
def chi_square_test(contingency_table: np.ndarray, alpha: float = 0.05) -> dict:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 4),
        "degrees_of_freedom": int(dof),
        "significant": bool(p_value < alpha)
    }


# Computes a confidence interval for the difference in proportions between treatment and control.
# Uses the Wilson method via statsmodels for accuracy at small proportions.
def confidence_interval(df: pd.DataFrame, outcome_col: str, alpha: float = 0.05) -> dict:
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    control_rate = float(control[outcome_col].mean())
    treatment_rate = float(treatment[outcome_col].mean())
    observed_lift = treatment_rate - control_rate

    control_ci = smp.proportion_confint(
        control[outcome_col].sum(), len(control), alpha=alpha, method="wilson"
    )
    treatment_ci = smp.proportion_confint(
        treatment[outcome_col].sum(), len(treatment), alpha=alpha, method="wilson"
    )

    ci_low = float(treatment_ci[0] - control_ci[1])
    ci_high = float(treatment_ci[1] - control_ci[0])

    return {
        "control_rate": round(control_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "observed_lift": round(observed_lift, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4)
    }


# Computes a two-sample t-test on conversion rates between treatment and control.
# Applied at Stage 3 to provide a frequentist complement to the chi-square test.
def t_test(df: pd.DataFrame, outcome_col: str, alpha: float = 0.05) -> dict:
    control_outcomes = df[df["group"] == "control"][outcome_col]
    treatment_outcomes = df[df["group"] == "treatment"][outcome_col]

    t_stat, p_value = stats.ttest_ind(control_outcomes, treatment_outcomes)

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha)
    }


# Computes a Bayesian estimate of the lift between treatment and control using Beta distributions.
# Models each group's conversion rate as a Beta(alpha, beta) posterior and estimates
# the probability that the treatment rate exceeds the control rate via Monte Carlo sampling.
def bayesian_comparison(df: pd.DataFrame, outcome_col: str, n_samples: int = 100_000) -> dict:
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    control_successes = control[outcome_col].sum()
    control_failures = len(control) - control_successes
    treatment_successes = treatment[outcome_col].sum()
    treatment_failures = len(treatment) - treatment_successes

    control_samples = np.random.beta(control_successes + 1, control_failures + 1, n_samples)
    treatment_samples = np.random.beta(treatment_successes + 1, treatment_failures + 1, n_samples)

    prob_treatment_wins = float((treatment_samples > control_samples).mean())

    return {
        "prob_treatment_wins": round(prob_treatment_wins, 4),
        "expected_lift": round(float((treatment_samples - control_samples).mean()), 4)
    }


# Applies Bonferroni correction to a list of p-values across multiple hypothesis tests.
# Controls the family-wise error rate by adjusting the significance threshold.
def bonferroni_correction(p_values: list, alpha: float = 0.05) -> dict:
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    adjusted_significance = [bool(p < adjusted_alpha) for p in p_values]
    cleaned_p_values = [float(p) for p in p_values]

    return {
        "original_p_values": cleaned_p_values,
        "adjusted_alpha": round(adjusted_alpha, 4),
        "significant": adjusted_significance
    }


# Orchestrates the full statistical analysis across all three funnel stages.
# Loads simulated data and applies chi-square, confidence interval, t-test,
# Bayesian comparison, and Bonferroni correction across adoption, re-engagement, and conversion.
def run_analysis(data_path: str = None) -> dict:
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "..", "data", "simulated_users.csv")
    df = pd.read_csv(data_path)

    # Stage 1: Feature Adoption
    adoption_table = build_contingency_table(df, "adopted")
    adoption_chi2 = chi_square_test(adoption_table)
    adoption_ci = confidence_interval(df, "adopted")
    adoption_bayes = bayesian_comparison(df, "adopted")

    # Stage 2: Notification Re-engagement
    reengagement_table = build_contingency_table(df, "reengaged")
    reengagement_chi2 = chi_square_test(reengagement_table)
    reengagement_ci = confidence_interval(df, "reengaged")
    reengagement_bayes = bayesian_comparison(df, "reengaged")

    # Stage 3: Premium Conversion
    conversion_table = build_contingency_table(df, "converted")
    conversion_chi2 = chi_square_test(conversion_table)
    conversion_ci = confidence_interval(df, "converted")
    conversion_ttest = t_test(df, "converted")
    conversion_bayes = bayesian_comparison(df, "converted")

    # Multiple testing correction across all three stages
    p_values = [adoption_chi2["p_value"], reengagement_chi2["p_value"], conversion_chi2["p_value"]]
    correction = bonferroni_correction(p_values)

    return {
        "adoption": {**adoption_chi2, **adoption_ci, **adoption_bayes},
        "reengagement": {**reengagement_chi2, **reengagement_ci, **reengagement_bayes},
        "conversion": {**conversion_chi2, **conversion_ci, **conversion_ttest, **conversion_bayes},
        "bonferroni": correction
    }


if __name__ == "__main__":
    results = run_analysis()
    for stage, metrics in results.items():
        print(f"\n── {stage.upper()} ──")
        for key, value in metrics.items():
            print(f"  {key}: {value}")