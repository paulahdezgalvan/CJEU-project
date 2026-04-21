"""Analyze whether government political position is associated with case outcomes.

EN: This script downloads the original CSV, keeps comparable outcomes
(`plaintiff` vs `commission`), and estimates the relationship between
`Gov_Left_Right` and the probability that the plaintiff wins.

ES: Este script descarga el CSV original, conserva resultados comparables
(`plaintiff` vs `commission`) y estima la relacion entre `Gov_Left_Right`
y la probabilidad de victoria del demandante.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats


DATA_URL = "https://raw.githubusercontent.com/ArthurDyevre/EU_Competition_Law/main/eu_competition_law.csv"
OUTPUT_DIR = Path("political_position_plots")


@dataclass
class TestResult:
    name: str
    value: float
    p_value: float


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    # EN: The source file uses Latin-1 encoding.
    # ES: El archivo fuente usa codificacion Latin-1.
    return pd.read_csv(url, sep=";", encoding="latin1")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # EN: Validate required columns before any transformation.
    # ES: Validamos columnas requeridas antes de transformar datos.
    required = ["Outcome", "Gov_Left_Right", "YEAR", "Area", "Legal_Tradition", "Legal_Practice"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    # EN: Keep only binary, comparable outcomes.
    # ES: Mantenemos solo resultados binarios y comparables.
    data["Outcome"] = data["Outcome"].astype(str).str.strip().str.lower()
    data = data[data["Outcome"].isin(["plaintiff", "commission"])].copy()

    # EN: Convert analytical fields to numeric and remove invalid rows.
    # ES: Convertimos campos analiticos a numericos y quitamos filas invalidas.
    data["Gov_Left_Right"] = pd.to_numeric(data["Gov_Left_Right"], errors="coerce")
    data["YEAR"] = pd.to_numeric(data["YEAR"], errors="coerce")
    data["Econ_Ideology_Median"] = pd.to_numeric(data["Econ_Ideology_Median"], errors="coerce")
    data["Legal_Expertise_Median"] = pd.to_numeric(data["Legal_Expertise_Median"], errors="coerce")
    data["Europhilia_Median"] = pd.to_numeric(data["Europhilia_Median"], errors="coerce")
    data["Ease_Doing_Business"] = pd.to_numeric(data["Ease_Doing_Business"], errors="coerce")
    data = data.dropna(subset=["Gov_Left_Right", "YEAR", "Outcome"])

    # EN: Binary dependent variable for logistic regression.
    # ES: Variable dependiente binaria para regresion logistica.
    data["plaintiff_win"] = (data["Outcome"] == "plaintiff").astype(int)
    return data


def summarize_groups(data: pd.DataFrame) -> pd.DataFrame:
    # EN: Descriptive stats by outcome group.
    # ES: Estadisticas descriptivas por grupo de resultado.
    return (
        data.groupby("Outcome")
        .agg(
            n=("Gov_Left_Right", "size"),
            mean_gov_left_right=("Gov_Left_Right", "mean"),
            median_gov_left_right=("Gov_Left_Right", "median"),
            std_gov_left_right=("Gov_Left_Right", "std"),
        )
        .reset_index()
    )


def run_group_tests(data: pd.DataFrame) -> list[TestResult]:
    # EN: Compare political-position distributions across outcome groups.
    # ES: Comparamos distribuciones de posicion politica entre resultados.
    plaintiff = data.loc[data["Outcome"] == "plaintiff", "Gov_Left_Right"].to_numpy()
    commission = data.loc[data["Outcome"] == "commission", "Gov_Left_Right"].to_numpy()

    welch = stats.ttest_ind(plaintiff, commission, equal_var=False, nan_policy="omit")
    mann_whitney = stats.mannwhitneyu(plaintiff, commission, alternative="two-sided")

    return [
        TestResult("Welch t-test", float(welch.statistic), float(welch.pvalue)),
        TestResult("Mann-Whitney U", float(mann_whitney.statistic), float(mann_whitney.pvalue)),
    ]


def run_logit(data: pd.DataFrame):
    # EN: Baseline model controlling only for year.
    # ES: Modelo base controlando solo por año.
    model = smf.logit("plaintiff_win ~ Gov_Left_Right + YEAR", data=data).fit(disp=False)
    coef = float(model.params["Gov_Left_Right"])
    conf = model.conf_int().loc["Gov_Left_Right"]
    return model, coef, conf


def run_adjusted_logit(data: pd.DataFrame):
    # EN: Richer specification with legal and contextual controls.
    # ES: Especificacion mas completa con controles legales y contextuales.
    formula = (
        "plaintiff_win ~ Gov_Left_Right + YEAR + Econ_Ideology_Median + "
        "Legal_Expertise_Median + Europhilia_Median + Ease_Doing_Business + "
        "C(Area) + C(Legal_Tradition) + C(Legal_Practice)"
    )
    model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
    coef = float(model.params["Gov_Left_Right"])
    conf = model.conf_int().loc["Gov_Left_Right"]
    return model, coef, conf


def build_summary_tables(data: pd.DataFrame, basic_model, basic_coef: float, basic_conf, adjusted_model, adjusted_coef: float, adjusted_conf) -> dict[str, pd.DataFrame]:
    # EN: Build report-friendly tables for export.
    # ES: Construimos tablas listas para reporte.
    group_summary = summarize_groups(data)

    plaintiff = data.loc[data["Outcome"] == "plaintiff", "Gov_Left_Right"].to_numpy()
    commission = data.loc[data["Outcome"] == "commission", "Gov_Left_Right"].to_numpy()
    welch = stats.ttest_ind(plaintiff, commission, equal_var=False, nan_policy="omit")
    mann_whitney = stats.mannwhitneyu(plaintiff, commission, alternative="two-sided")

    tests_summary = pd.DataFrame(
        [
            {"test": "Welch t-test", "statistic": float(welch.statistic), "p_value": float(welch.pvalue)},
            {"test": "Mann-Whitney U", "statistic": float(mann_whitney.statistic), "p_value": float(mann_whitney.pvalue)},
        ]
    )

    model_summary = pd.DataFrame(
        [
            {
                "model": "Logit adjusted by YEAR",
                "term": "Gov_Left_Right",
                "coef": basic_coef,
                "odds_ratio": math.exp(basic_coef),
                "ci_low": math.exp(float(basic_conf.iloc[0])),
                "ci_high": math.exp(float(basic_conf.iloc[1])),
                "p_value": float(basic_model.pvalues["Gov_Left_Right"]),
            },
            {
                "model": "Logit with controls",
                "term": "Gov_Left_Right",
                "coef": adjusted_coef,
                "odds_ratio": math.exp(adjusted_coef),
                "ci_low": math.exp(float(adjusted_conf.iloc[0])),
                "ci_high": math.exp(float(adjusted_conf.iloc[1])),
                "p_value": float(adjusted_model.pvalues["Gov_Left_Right"]),
            },
        ]
    )

    return {
        "group_summary": group_summary,
        "tests_summary": tests_summary,
        "model_summary": model_summary,
    }


def export_summary_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    # EN: Export both machine-friendly (CSV) and presentation-friendly (HTML) tables.
    # ES: Exportamos tablas para maquina (CSV) y presentacion (HTML).
    output_dir.mkdir(exist_ok=True)
    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        html_path = output_dir / f"{name}.html"
        table.to_csv(csv_path, index=False)
        table.to_html(html_path, index=False, border=0)


def make_plots(data: pd.DataFrame, output_dir: Path) -> None:
    # EN: Save visuals to disk for quick interpretation and reporting.
    # ES: Guardamos visualizaciones para interpretacion y reporte rapido.
    output_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="Outcome", y="Gov_Left_Right", hue="Outcome", ax=ax, palette="Set2", legend=False)
    sns.stripplot(
        data=data,
        x="Outcome",
        y="Gov_Left_Right",
        ax=ax,
        color="black",
        alpha=0.25,
        size=2,
        jitter=0.2,
    )
    ax.set_title("Distribution of political position by case outcome")
    ax.set_xlabel("Case outcome")
    ax.set_ylabel("Gov_Left_Right")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_gov_left_right_by_outcome.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=data,
        x="Gov_Left_Right",
        y="plaintiff_win",
        logistic=True,
        scatter_kws={"alpha": 0.15, "s": 18},
        line_kws={"color": "crimson", "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Estimated probability of plaintiff win")
    ax.set_xlabel("Gov_Left_Right")
    ax.set_ylabel("Probability of plaintiff_win")
    fig.tight_layout()
    fig.savefig(output_dir / "probability_plaintiff_win_vs_gov_left_right.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    # EN: Allow configurable output folder and data URL for portable execution.
    # ES: Permite configurar carpeta de salida y URL de datos para ejecucion portable.
    parser = argparse.ArgumentParser(description="Political position vs case outcome analysis")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where plots and summary tables are saved",
    )
    parser.add_argument(
        "--data-url",
        default=DATA_URL,
        help="CSV URL to analyze",
    )
    return parser.parse_args()


def main() -> None:
    # EN: Main pipeline: load, clean, model, plot, and export.
    # ES: Flujo principal: cargar, limpiar, modelar, graficar y exportar.
    args = parse_args()
    output_dir = Path(args.output_dir)

    df = load_data(args.data_url)
    data = prepare_data(df)

    make_plots(data, output_dir)

    print("Analyzed cases:", len(data))
    print()
    print(f"Plots saved in: {output_dir.resolve()}")
    print()
    print("Summary by outcome:")
    print(summarize_groups(data).to_string(index=False))
    print()

    print("Group difference tests:")
    for result in run_group_tests(data):
        print(f"- {result.name}: statistic={result.value:.4f}, p={result.p_value:.6f}")
    print()

    model, coef, conf = run_logit(data)
    odds_ratio = math.exp(coef)
    ci_low = math.exp(float(conf.iloc[0]))
    ci_high = math.exp(float(conf.iloc[1]))

    print("Logistic regression adjusted by YEAR:")
    print(f"- Coefficient Gov_Left_Right: {coef:.6f}")
    print(f"- Odds ratio: {odds_ratio:.4f}")
    print(f"- 95% CI OR: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"- p-value: {model.pvalues['Gov_Left_Right']:.6f}")
    print()

    adjusted_model, adjusted_coef, adjusted_conf = run_adjusted_logit(data)
    adjusted_or = math.exp(adjusted_coef)
    adjusted_ci_low = math.exp(float(adjusted_conf.iloc[0]))
    adjusted_ci_high = math.exp(float(adjusted_conf.iloc[1]))

    print("Logistic regression with controls:")
    print(f"- Coefficient Gov_Left_Right: {adjusted_coef:.6f}")
    print(f"- Odds ratio: {adjusted_or:.4f}")
    print(f"- 95% CI OR: [{adjusted_ci_low:.4f}, {adjusted_ci_high:.4f}]")
    print(f"- p-value: {adjusted_model.pvalues['Gov_Left_Right']:.6f}")
    print()

    tables = build_summary_tables(data, model, coef, conf, adjusted_model, adjusted_coef, adjusted_conf)
    export_summary_tables(tables, output_dir)
    print(f"Tables exported to: {output_dir.resolve()}")
    print()

    if adjusted_model.pvalues["Gov_Left_Right"] < 0.05:
        direction = "increases" if adjusted_coef > 0 else "decreases"
        conclusion = (
            "There is statistical evidence of association: as government ideology "
            f"moves to the right, the probability of plaintiff victory {direction}."
        )
    else:
        conclusion = (
            "There is no statistically significant evidence that government political "
            "position affects case outcomes in this dataset."
        )

    print(textwrap.fill(conclusion, width=88))


if __name__ == "__main__":
    main()