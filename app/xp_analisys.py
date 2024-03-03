import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go


def get_input_test_stats(df, group_col, label_col, sample_size):
    conversion_rates = pd.concat(
        [
            df.groupby(group_col)[label_col].apply("mean").rename("CVR"),
            df.groupby(group_col)[label_col].apply("std").rename("STD"),
            df.groupby(group_col)[label_col]
            .apply(lambda x: round(np.std(x) / np.sqrt(sample_size), 3))
            .rename("STDErr"),
            df.groupby(group_col)[label_col].apply("count").rename("count"),
            df.groupby(group_col)[label_col].apply("sum").rename("converted"),
        ],
        axis=1,
    )
    return conversion_rates


def get_ztats_pvalue(m_0, m_1, std_0, std_1, count_0, count_1, alpha=0.05):
    standard_norm = stats.norm(0, 1)
    z_alpha = standard_norm.ppf(alpha)
    z_alpha
    Zstat = (m_0 - m_1) / np.sqrt((pow(std_1, 2) / count_1) + (pow(std_0, 2) / count_0))
    p_value = stats.norm.sf(abs(Zstat)) * 2
    return Zstat, p_value


def get_confidence_interval(m_0, count_0):
    margin_95confidence = np.sqrt((m_0 * (1 - m_0)) / count_0) * 2
    CI = [round(m_0 - margin_95confidence, 3), round(m_0 + margin_95confidence, 3)]
    return CI, margin_95confidence


def get_all_stats_two_sided_test(
    comb, m_0, m_1, std_0, std_1, count_0, count_1, alpha=0.05, st=None
):
    control, treatment = comb
    zstat, p_value = get_ztats_pvalue(m_0, m_1, std_0, std_1, count_0, count_1)

    CI_0 = get_confidence_interval(m_0, count_0)
    CI_1 = get_confidence_interval(m_1, count_1)
    print(f"z-stat: {zstat}")
    print(f"p-value: {p_value}")
    print(f"CI 95% for {control} group : {CI_0[0]}")
    print(f"CI 95% for {treatment} group : {CI_1[0]}")

    if p_value >= alpha or p_value <= -alpha:
        print(f"For a confidence of {1- alpha}% we can not reject the Null hypothesis")
        print(
            "There is no evidence that Control and Treatment distributions are not the same."
        )
    else:
        print(
            f"For a confidence of {1- alpha}% we reject the Null hypothesis, Control and Treatment distributions are different"
        )

    if st:
        st.text(f"z-stat: {zstat}")
        st.text(f"p-value: {p_value}")
        st.text(f"CI 95% for {control} group : {CI_0[0]}")
        st.text(f"CI 95% for {treatment} group : {CI_1[0]}")

        if p_value >= alpha or p_value <= -alpha:
            st.text(
                f"For a confidence of {1- alpha}% we can not reject the Null hypothesis"
            )
            st.text(
                "There is no evidence that Control and Treatment distributions are not the same."
            )
        else:
            st.text(
                f"For a confidence of {1- alpha}% we reject the Null hypothesis, Control and Treatment distributions are different"
            )


def visualise_distribution(**kwargs):
    fig = go.Figure()
    min_x = min([r.mu - 4 * r.sigma for r in kwargs.values()])
    max_x = max([r.mu + 4 * r.sigma for r in kwargs.values()])
    x = np.arange(min_x, max_x, 0.001)
    for key, value in kwargs.items():
        y = stats.norm.pdf(x, value.mu, value.sigma)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name=key))
        fig.add_vline(
            x=2 * value.sigma,
            line_width=1,
            annotation_text=f"{key}:95% confidence",
            line_dash="dash",
            line_color="green",
        )
    return fig
