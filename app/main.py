import plotly.express as px
import streamlit as st
import numpy as np
from scipy import stats
import itertools
from xp_analisys import (
    get_input_test_stats,
    get_all_stats_two_sided_test,
    visualise_distribution,
)

df = px.data.gapminder()

fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)


def define_squeared_delta(metric_baseline, pct_lift):
    """
    Calculate delta square base on:
    - Mean of Metric Value: example, currently CTR is 15% = (0.15)
    - Minimun Detectect Effect: example we expecte a lift of 20%
    """
    print("pct_lift", pct_lift)
    print("metric_baseline", metric_baseline)
    metric_treatment = metric_baseline * (1.0 + (pct_lift / 100.0))

    print("Mean H1", metric_treatment)
    print(
        "pow(metric_baseline - metric_treatment, 2)",
        pow(metric_baseline - metric_treatment, 2),
    )
    return (metric_baseline - metric_treatment) ** 2


def z_critical_values(alpha=0.05, beta=0.2):
    """
    It return part of numerator term related with the z scores
    of constant alpha and beta:
    pow(Z_(1 - alpha/2) + Z_(1-beta) , 2)
    """
    standard_norm = stats.norm(0, 1)

    z_alpha = standard_norm.ppf(1 - (alpha / 2.0))

    statistical_power = 1 - beta
    z_power = standard_norm.ppf(statistical_power)
    print("alpha", alpha)
    print("beta", beta)
    print("power", statistical_power)
    print("z_(1- alpha/2)", z_alpha)
    print("z_power", z_power)
    print("pow(z_alpha + z_power, 2)", pow(z_alpha + z_power, 2))

    return pow(z_alpha + z_power, 2)



def aprox_two_sided_test(p):
    variance = p * (1 - p)
    print(variance)
    return 2 * variance  # 2.0 * (sigma**2)


def estimated_sample_size(mean_metric, expected_pp_lift, alpha=0.05, beta=0.2):
    numerator = z_critical_values(alpha=alpha, beta=beta) * aprox_two_sided_test(
        mean_metric
    )
    denominator = (expected_pp_lift / 100.0) ** 2
    print(f"H1 mean: {(mean_metric+ (expected_pp_lift/100.0))}")
    return numerator / denominator


tab1, tab2 = st.tabs(
    [
        "A/B test Sample size",
        "A/B test See XP",
    ]
)
with tab1:
    st.header("Sample size")
    alpha = st.number_input("Alpha", value=0.05)
    beta = st.number_input("Beta", value=0.2)
    mean_baseline_metric = st.number_input(
        "Mean of Baseline Metric", key="baseline_metric"
    )
    pp_lift = st.number_input("Expected lift in percentage point (pp)", value=1.0)

    if alpha and beta and mean_baseline_metric and pp_lift:  # and sigma:
        sample_size = estimated_sample_size(
            mean_metric=mean_baseline_metric,
            expected_pp_lift=pp_lift,
            beta=beta,
            alpha=alpha,
        )
        total_sample_size = 2 * sample_size
        # st.text(f"Detectable metric in treatment: {pp_lift + mean_baseline_metric}")
        st.text(f"Sample size by group: {int(sample_size)}")
        st.text(f"Total Sample size: {int(total_sample_size)}")

    st.latex(
        r"""
        \frac
        {(Z_{1 - \alpha/2} + Z_{1 - \beta})^2 * (\sigma_{0}^2 + \sigma_{1}^2) }
        {(\mu_{0} - \mu_{1})^2}
    """
    )
    st.latex(
        r"""
        \sigma_{0}^2 = ~ \mu_{0} * (1 - \mu_{0})
    """
    )
    st.latex(
        r"""
                \text{ We are assuming  } \sigma_{0} = \sigma_{1}
    """
    )
    st.latex(
        r"""
        \mu_{0} = \text{ Mean of metric of baseline }
    """
    )
    st.latex(
        r"""
                \mu_{1} = \mu_{0} + pp. \\
                \text{ Expected lift pct to be detected }
    """
    )
    st.text_input("XP name")
    st.text_area("Xp description")

with tab2:

    def click_button():
        st.session_state.dataframe = st.session_state.dataframe.drop_duplicates(
            st.session_state.dedup_column, keep=False
        )
        st.session_state.dedup_clicked = True

    import pandas as pd

    st.header("XP results")
    uploaded_file = st.file_uploader("Upload youd file here")

    if uploaded_file is not None:
        if "dataframe" not in st.session_state:

            st.session_state.dataframe = pd.read_csv(uploaded_file)
            st.session_state.data_loaded = True

        st.text("Dataset sample")
        st.write(st.session_state.dataframe.sample(10))
        st.text(f"DF size: {st.session_state.dataframe.shape[0]}")

    if "data_loaded" in st.session_state:
        columns = st.session_state.dataframe.columns.to_list()
        st.session_state.dedup_column = st.selectbox(
            "Deduplicate vents by column:", columns
        )
        st.button("Deduplicate", on_click=click_button)

    if "dedup_clicked" not in st.session_state:
        st.session_state.dedup_clicked = False

    if st.session_state.dedup_clicked:
        # T he message and nested widget will remain on the page
        st.text(f"Deduplicated: size: {st.session_state.dataframe.shape[0]}")

    if st.session_state.dedup_clicked:
        selected_column = st.selectbox("Pick the group column", columns)
        if st.session_state.dataframe[selected_column].nunique() > 3:
            st.text("Too many groups to analyse, please pick the right column")
        else:
            alpha_xp = st.number_input("Alpha", value=0.05, key="alpha_xp")
            xp_groups = st.session_state.dataframe[selected_column].unique().tolist()
            st.text(xp_groups)
            sample_size = st.number_input("Choose sample size by group", min_value=31)

            group_data = {}
            for group in xp_groups:
                group_data[group] = (
                    st.session_state.dataframe[
                        st.session_state.dataframe[selected_column] == group
                    ]
                    .sample(frac=1, random_state=22)
                    .reset_index(drop=True)
                    .head(sample_size)
                )
                st.text(f"{group}: {group_data[group].shape}")
            st.session_state.groups_selection = group_data

    if "dedup_clicked" in st.session_state and "groups_selection" in st.session_state:

        def click_start_test_button():
            ab_test = pd.concat(
                [*list(st.session_state.groups_selection.values())], axis=0
            )
            ab_test.reset_index(drop=True, inplace=True)

            st.session_state.ab_test = ab_test
            # st.text(f"AB Test dataset: {ab_test.shape}")
            st.session_state.start_test_clicked = True

        st.button("Start Test", on_click=click_start_test_button)

        if "start_test_clicked" not in st.session_state:
            st.session_state.start_test_clicked = False

    if "start_test_clicked" in st.session_state and st.session_state.start_test_clicked:

        st.header("Results", divider="blue")
        st.text(f"sample_size: {sample_size}")

        groups_names = st.session_state.ab_test[selected_column].unique().tolist()
        combinations = list(itertools.combinations(groups_names, r=2))

        for comb in combinations:
            control, treatment = comb
            st.header(f"Results for groups: {control} and {treatment}", divider="blue")
            df_comb = st.session_state.ab_test[
                st.session_state.ab_test[selected_column].isin(list(comb))
            ]

            input_stats_df = get_input_test_stats(
                df_comb,
                group_col=selected_column,
                label_col="converted",
                sample_size=sample_size,
            )
            st.text(input_stats_df)
            input_stats = input_stats_df.T.to_dict()

            # Todo generalize this call
            get_all_stats_two_sided_test(
                comb,
                input_stats[control]["CVR"],
                input_stats[treatment]["CVR"],
                input_stats[control]["STD"],
                input_stats[treatment]["STD"],
                input_stats[control]["count"],
                input_stats[treatment]["count"],
                alpha=0.05,
                st=st,
            )
            from collections import namedtuple

            input_visualization = {}
            Rating = namedtuple("Rating", ["mu", "sigma"])
            
            for group in [control, treatment]:
                input_visualization[group] = Rating(
                    input_stats[group]["CVR"], input_stats[group]["STD"]
                )
            fig = visualise_distribution(**input_visualization)
            st.plotly_chart(fig)
            st.divider()
