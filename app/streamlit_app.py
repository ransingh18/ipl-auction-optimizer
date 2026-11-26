"""
IPL Auction Optimizer — Streamlit Dashboard
Week 5 | BUSAN302 × BUSAN300 Portfolio Project | Ranveer Singh

Multi-page app:
  🏏 Overview        — project summary + key metrics
  💰 Price Predictor  — explore predicted vs actual prices
  🏆 Squad Optimizer  — build constrained squads with ILP
  🧠 Bias Analysis    — interactive behavioural bias explorer
  📊 Player Explorer  — search, filter, compare players
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import sys, os

# ── Path setup ──
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))
from optimizer import (
    prepare_player_pool, optimise_squad, compute_performance_score,
    compute_value_score, IPL_RULES
)

# ── Page config ──
st.set_page_config(
    page_title="IPL Auction Optimizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 5px 0 0 0; opacity: 0.85; font-size: 0.9rem; }
    .bias-strong { color: #e74c3c; font-weight: bold; }
    .bias-weak { color: #95a5a6; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data/processed/predictions_with_bias.csv'))
    return df

@st.cache_data
def load_model_metadata():
    with open(os.path.join(PROJECT_DIR, 'models/model_metadata.json')) as f:
        return json.load(f)

@st.cache_resource
def load_model():
    with open(os.path.join(PROJECT_DIR, 'models/best_model.pkl'), 'rb') as f:
        return pickle.load(f)

df = load_data()
meta = load_model_metadata()


# ═══════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════
st.sidebar.title("🏏 IPL Auction Optimizer")
st.sidebar.markdown("*BUSAN302 × BUSAN300*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📌 Executive Summary", "🏠 Overview", "💰 Price Predictor", "🏆 Squad Optimizer",
     "🧠 Bias Analysis", "📊 Player Explorer", "💡 Strategy", "🔭 What's Next"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by** Ranveer Singh  \n"
    "BCom — Finance & Business Analytics  \n"
    "University of Auckland"
)


# ═══════════════════════════════════════════════
# PAGE 0: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════
if page == "📌 Executive Summary":
    st.title("📌 Executive Summary")
    st.markdown(
        "*A business-language overview of what this tool does, what it found, "
        "and why it matters for franchise decision-making.*"
    )

    st.markdown("---")

    # ── The headline numbers ──
    st.subheader("The Bottom Line")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="metric-card">
            <h2>₹297 Cr</h2>
            <p>Total overspend identified<br>across 12 auction cycles</p>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h2>₹594 Cr</h2>
            <p>Total pricing inefficiency<br>captured by the model</p>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
            <h2>78%</h2>
            <p>Mega-auction price premium<br>over mini auctions</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── What this tool does ──
    st.subheader("What This Tool Does")
    st.markdown("""
    This system combines **machine learning** and **integer linear programming**
    to give IPL franchises a data-driven edge at the auction table:

    | Module | What It Does | Business Value |
    |--------|-------------|----------------|
    | **Price Predictor** | Estimates fair price from 37 performance & market features | Know when you're overpaying *before* you bid |
    | **Squad Optimizer** | Builds the best squad under real constraints (₹120 Cr budget, 8 overseas, role minimums) | Maximise squad strength per rupee |
    | **Bias Detector** | Quantifies 6 cognitive biases with statistical tests | Exploit competitors' systematic mistakes |
    """)

    st.markdown("---")

    # ── Three findings that matter ──
    st.subheader("Three Findings That Matter")

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("#### 1. Base Price Anchoring Costs Franchises Crores")
        st.markdown(
            "The base price set by the BCCI explains **20% of final sale price** "
            "beyond all performance stats combined. That means franchises are "
            "psychologically anchored to an arbitrary starting number — "
            "a textbook example of Kahneman & Tversky's anchoring effect "
            "playing out in a ₹1,000+ Cr market."
        )
        # Mini chart: anchoring
        has_base = df[df['base_price_cr'] > 0]
        fig_anchor = px.scatter(has_base, x='base_price_cr', y='sold_price_cr',
                                opacity=0.3, trendline='ols',
                                labels={'base_price_cr': 'Base Price (₹ Cr)',
                                        'sold_price_cr': 'Sale Price (₹ Cr)'},
                                color_discrete_sequence=['#667eea'])
        fig_anchor.update_layout(height=300, margin=dict(t=10, b=10),
                                 showlegend=False)
        st.plotly_chart(fig_anchor, use_container_width=True)

    with f2:
        st.markdown("#### 2. The ₹5.21 Cr Problem")
        st.markdown(
            "The 2025 mega auction averaged **₹5.21 Cr per player** — "
            "the highest in IPL history. When 10 teams rebuild simultaneously, "
            "competitive bidding inflates prices 78% above mini-auction levels. "
            "A franchise that recognised this structural inflation could have "
            "saved ₹2-3 Cr per marquee signing."
        )
        yearly = df.groupby('auction_year')['sold_price_cr'].mean().reset_index()
        mega_years = [2014, 2018, 2022, 2025]
        yearly['type'] = yearly['auction_year'].apply(
            lambda y: 'Mega' if y in mega_years else 'Mini')
        fig_yr = px.bar(yearly, x='auction_year', y='sold_price_cr', color='type',
                        color_discrete_map={'Mega': '#eb3349', 'Mini': '#3498db'},
                        labels={'sold_price_cr': 'Avg Price (₹ Cr)', 'auction_year': ''},
                        text_auto='.1f')
        fig_yr.update_layout(height=300, margin=dict(t=10, b=10),
                             showlegend=False)
        st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown("#### 3. The Mid-Tier Sweet Spot")
    st.markdown(
        "Players with **3-5 Player of the Match awards** average ₹5.68 Cr — "
        "80% of the performance signal of elite 6+ POTM players (₹7.10 Cr) "
        "at a 20% discount. A franchise targeting mid-tier stars over marquee names "
        "could build a stronger squad for less, exploiting the market's tendency "
        "to overpay for star power."
    )

    st.markdown("---")

    # ── Methods at a glance ──
    st.subheader("Methods at a Glance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**Data**")
        st.markdown("1,279 player-auctions across 12 IPL seasons (2013-2025)")
    with m2:
        st.markdown("**Models**")
        st.markdown(
            "Gradient Boosting (best), plus Ridge, Random Forest, XGBoost. "
            "Temporal train/test split to prevent data leakage."
        )
    with m3:
        st.markdown("**Optimization**")
        st.markdown(
            "Integer Linear Programming (PuLP + HiGHS solver) with real IPL "
            "constraints: budget, roster size, overseas cap, role minimums."
        )

    st.markdown("---")
    st.markdown(
        "*Built by **Ranveer Singh** · BCom (Finance & Business Analytics) · "
        "University of Auckland · 2025*"
    )


# ═══════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════
elif page == "🏠 Overview":
    st.title("🏏 IPL Auction Optimizer")
    st.markdown(
        "Predict player auction values, detect behavioural biases, "
        "and build optimal squads under real IPL constraints."
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Players Analysed", f"{len(df):,}")
    with col2:
        st.metric("Auction Years", f"{df['auction_year'].nunique()}")
    with col3:
        st.metric("ML Features", f"{len(meta['feature_names'])}")
    with col4:
        best = meta['best_model']
        rmse = meta['results'][best]['rmse']
        st.metric("Best Model RMSE", f"₹{rmse:.2f} Cr")

    st.markdown("---")

    # Model comparison
    st.subheader("ML Model Performance")
    results_df = pd.DataFrame(meta['results']).T
    results_df.index.name = 'Model'
    results_df = results_df.reset_index()

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Test RMSE (₹ Cr) — lower is better', 'Cross-Validation R² — higher is better'])

    fig.add_trace(go.Bar(
        x=results_df['Model'], y=results_df['rmse'],
        marker_color=['#e74c3c' if m == best else '#3498db' for m in results_df['Model']],
        name='RMSE', showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=results_df['Model'], y=results_df['cv_r2'],
        marker_color=['#e74c3c' if m == best else '#2ecc71' for m in results_df['Model']],
        name='CV R²', showlegend=False
    ), row=1, col=2)

    fig.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Top 15 Features")
    fi = pd.DataFrame(meta['feature_importance'], columns=['Feature', 'Importance']).head(15)
    fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis')
    fig_fi.update_layout(height=450, yaxis=dict(autorange='reversed'),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Project timeline
    st.subheader("Project Timeline")
    timeline = pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
        'Deliverable': [
            'Data pipeline & feature engineering (37 features)',
            'EDA & ML models (GBM best: RMSE ₹5.18)',
            'Squad optimizer (ILP with 4 scenarios)',
            'Behavioural bias analysis (6 biases tested)',
            'Streamlit web app (this dashboard)',
            'GitHub repo & portfolio packaging'
        ],
        'Status': ['✅', '✅', '✅', '✅', '✅', '✅']
    })
    st.dataframe(timeline, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 2: PRICE PREDICTOR
# ═══════════════════════════════════════════════
elif page == "💰 Price Predictor":
    st.title("💰 Price Prediction Analysis")

    # Year filter
    years = sorted(df['auction_year'].unique())
    selected_years = st.multiselect("Filter by Auction Year", years, default=years)
    filtered = df[df['auction_year'].isin(selected_years)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted vs Actual Prices")
        fig = px.scatter(
            filtered, x='sold_price_cr', y='predicted_price_cr',
            color='role', hover_name='player',
            hover_data=['auction_year', 'team', 'player_origin', 'base_price_cr'],
            labels={'sold_price_cr': 'Actual Price (₹ Cr)', 'predicted_price_cr': 'Predicted Price (₹ Cr)'},
            opacity=0.6
        )
        max_val = max(filtered['sold_price_cr'].max(), filtered['predicted_price_cr'].max()) + 1
        fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(dash='dash', color='gray'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price Distribution by Role")
        fig2 = px.box(
            filtered, x='role', y='sold_price_cr', color='role',
            labels={'sold_price_cr': 'Sold Price (₹ Cr)', 'role': 'Role'},
            points='outliers'
        )
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Most overvalued / undervalued
    st.subheader("Most Over & Under-Valued Players")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🔴 Most Overpaid** (actual >> predicted)")
        overpaid = filtered.nlargest(10, 'prediction_error')[
            ['player', 'auction_year', 'role', 'sold_price_cr', 'predicted_price_cr', 'prediction_error']
        ].copy()
        overpaid.columns = ['Player', 'Year', 'Role', 'Actual ₹Cr', 'Predicted ₹Cr', 'Overpay ₹Cr']
        st.dataframe(overpaid.style.format({
            'Actual ₹Cr': '₹{:.2f}', 'Predicted ₹Cr': '₹{:.2f}', 'Overpay ₹Cr': '₹{:.2f}'
        }), hide_index=True)

    with col_b:
        st.markdown("**🟢 Best Value** (actual << predicted)")
        underpaid = filtered.nsmallest(10, 'prediction_error')[
            ['player', 'auction_year', 'role', 'sold_price_cr', 'predicted_price_cr', 'prediction_error']
        ].copy()
        underpaid.columns = ['Player', 'Year', 'Role', 'Actual ₹Cr', 'Predicted ₹Cr', 'Saving ₹Cr']
        underpaid['Saving ₹Cr'] = underpaid['Saving ₹Cr'].abs()
        st.dataframe(underpaid.style.format({
            'Actual ₹Cr': '₹{:.2f}', 'Predicted ₹Cr': '₹{:.2f}', 'Saving ₹Cr': '₹{:.2f}'
        }), hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 3: SQUAD OPTIMIZER
# ═══════════════════════════════════════════════
elif page == "🏆 Squad Optimizer":
    st.title("🏆 Squad Optimizer")
    st.markdown("Build optimal IPL squads using Integer Linear Programming (ILP)")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.slider("Budget (₹ Cr)", 50.0, 150.0, 120.0, 5.0)
    with col2:
        objective = st.selectbox("Optimise for",
            ['composite_value', 'performance_score', 'value_score'],
            format_func=lambda x: {
                'composite_value': '⚖️ Balanced (Performance + Value)',
                'performance_score': '🏏 Pure Performance',
                'value_score': '💎 Best Value / Bargains'
            }[x])
    with col3:
        auction_year = st.selectbox("Auction Year", [2025, 2023, 2022], index=0)

    # Retained players
    year_players = df[df['auction_year'] == auction_year]['player'].tolist()
    retained = st.multiselect("Retain Players (auto-include)", year_players, default=[])

    if st.button("🚀 Optimise Squad", type="primary"):
        with st.spinner("Running ILP solver..."):
            predictions = pd.read_csv(os.path.join(PROJECT_DIR, 'data/processed/predictions.csv'))
            pool = prepare_player_pool(predictions, auction_year=auction_year)

            result = optimise_squad(
                pool, budget=budget, objective=objective,
                retained_players=retained if retained else None
            )

        if result['status'] == 'optimal' and len(result['squad']) > 0:
            squad = result['squad']

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Squad Size", result['total_players'])
            m2.metric("Budget Used", f"₹{result['budget_used']:.1f} Cr")
            m3.metric("Budget Left", f"₹{result['budget_remaining']:.1f} Cr")
            m4.metric("Overseas", f"{result['overseas_count']}/8")

            # Squad table
            st.subheader("Selected Squad")
            display_cols = ['player', 'role', 'player_origin', 'sold_price_cr',
                           'performance_score', 'value_score']
            available_cols = [c for c in display_cols if c in squad.columns]
            squad_display = squad[available_cols].sort_values('sold_price_cr', ascending=False)
            squad_display.columns = [c.replace('_', ' ').title() for c in available_cols]
            st.dataframe(
                squad_display.style.format({
                    'Sold Price Cr': '₹{:.2f}',
                    'Performance Score': '{:.1f}',
                    'Value Score': '{:.2f}'
                }),
                use_container_width=True, hide_index=True
            )

            # Role breakdown chart
            col_a, col_b = st.columns(2)
            with col_a:
                role_counts = squad['role'].value_counts()
                fig_role = px.pie(values=role_counts.values, names=role_counts.index,
                                title="Squad Composition by Role",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_role, use_container_width=True)

            with col_b:
                role_spend = squad.groupby('role')['sold_price_cr'].sum().reset_index()
                fig_spend = px.bar(role_spend, x='role', y='sold_price_cr',
                                  color='role', title="Budget Allocation by Role",
                                  labels={'sold_price_cr': 'Total Spend (₹ Cr)', 'role': 'Role'},
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                fig_spend.update_layout(showlegend=False)
                st.plotly_chart(fig_spend, use_container_width=True)
        else:
            st.error("No feasible solution found. Try adjusting budget or constraints.")


# ═══════════════════════════════════════════════
# PAGE 4: BIAS ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🧠 Bias Analysis":
    st.title("🧠 Behavioural Bias Analysis")
    st.markdown("How cognitive biases distort IPL auction pricing")

    bias_tab = st.radio("Select Bias", [
        "⚓ Anchoring", "⏰ Recency", "⭐ Star Premium",
        "🌍 Overseas Premium", "📈 Auction Inflation", "📋 Summary"
    ], horizontal=True)

    if bias_tab == "⚓ Anchoring":
        st.subheader("Anchoring Bias: Base Price → Sale Price")
        st.markdown(
            "**Finding: STRONG** (r = 0.61, p < 0.001) — Base price is the #1 predictor, "
            "explaining ~20% additional variance beyond all performance stats."
        )

        has_base = df[df['base_price_cr'] > 0]
        fig = px.scatter(has_base, x='base_price_cr', y='sold_price_cr',
                        color='role', hover_name='player',
                        hover_data=['auction_year', 'team'],
                        labels={'base_price_cr': 'Base Price (₹ Cr)',
                                'sold_price_cr': 'Sold Price (₹ Cr)'},
                        title="Base Price vs Sold Price", opacity=0.5)
        max_bp = has_base['base_price_cr'].max() + 0.5
        fig.add_shape(type='line', x0=0, y0=0, x1=max_bp, y1=max_bp,
                      line=dict(dash='dash', color='gray'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Multiplier table
        has_base_c = has_base.copy()
        has_base_c['base_bucket'] = pd.cut(has_base_c['base_price_cr'],
            bins=[0, 0.2, 0.5, 1.0, 1.5, 3],
            labels=['₹0.1-0.2', '₹0.2-0.5', '₹0.5-1.0', '₹1.0-1.5', '₹1.5+'])
        mult_df = has_base_c.groupby('base_bucket', observed=True).agg(
            avg_multiple=('price_multiple', 'mean'),
            median_multiple=('price_multiple', 'median'),
            players=('player', 'count')
        ).round(2)
        st.dataframe(mult_df, use_container_width=True)

    elif bias_tab == "⏰ Recency":
        st.subheader("Recency Bias: Does Recent Form Override Career?")
        st.markdown(
            "**Finding: NOT confirmed** — Career and recent stats show similar "
            "correlation with price. Franchises don't significantly overweight recent form."
        )
        exp = df[df['has_ipl_experience'] == 1]

        metrics_data = pd.DataFrame({
            'Metric': ['Recent Runs', 'Recent SR', 'Recent Wickets',
                       'Career Runs', 'Career SR', 'Career Wickets'],
            'Type': ['Recent']*3 + ['Career']*3,
            'Correlation': [
                exp[['recent_runs','sold_price_cr']].corr().iloc[0,1],
                exp[['recent_batting_sr','sold_price_cr']].corr().iloc[0,1],
                exp[['recent_wickets','sold_price_cr']].corr().iloc[0,1],
                exp[['career_runs','sold_price_cr']].corr().iloc[0,1],
                exp[['career_batting_sr','sold_price_cr']].corr().iloc[0,1],
                exp[['career_wickets','sold_price_cr']].corr().iloc[0,1],
            ]
        })
        fig = px.bar(metrics_data, x='Correlation', y='Metric', color='Type',
                    orientation='h', title="Correlation with Sold Price: Recent vs Career",
                    color_discrete_map={'Recent': '#e74c3c', 'Career': '#3498db'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif bias_tab == "⭐ Star Premium":
        st.subheader("Star / Halo Premium: POTM Awards → Price")
        st.markdown(
            "**Finding: STRONG pattern** — 6+ POTM players average ₹7.10 Cr vs "
            "₹1.59 Cr for unawarded (4.5×). Model captures this via `total_potm` feature."
        )

        star = df.groupby('star_tier', observed=True)['sold_price_cr'].agg(
            ['mean', 'median', 'count']).reset_index()
        fig = px.bar(star, x='star_tier', y=['mean', 'median'],
                    barmode='group', title='Average Price by Star Tier',
                    labels={'value': 'Price (₹ Cr)', 'star_tier': 'POTM Award Tier'},
                    color_discrete_sequence=['#9b59b6', '#d7bde2'])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Role pricing
        role_p = df.groupby('role')['sold_price_cr'].agg(['mean', 'median']).sort_values('mean').reset_index()
        fig2 = px.bar(role_p, x='role', y=['mean', 'median'], barmode='group',
                     title='Average Price by Role',
                     labels={'value': 'Price (₹ Cr)', 'role': 'Role'},
                     color_discrete_sequence=['#2ecc71', '#a9dfbf'])
        st.plotly_chart(fig2, use_container_width=True)

    elif bias_tab == "🌍 Overseas Premium":
        st.subheader("Overseas Premium: Foreign Player Tax")
        st.markdown(
            "**Finding: Visible but partly rational** — Overseas cost 48% more per 1,000 runs, "
            "but the 8-player cap creates artificial scarcity."
        )

        fig = px.histogram(df, x='sold_price_cr', color='player_origin',
                          barmode='overlay', nbins=40, opacity=0.6,
                          labels={'sold_price_cr': 'Sold Price (₹ Cr)',
                                  'player_origin': 'Origin'},
                          title="Price Distribution: Indian vs Overseas")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        origin_comp = df.groupby('player_origin').agg(
            avg_price=('sold_price_cr', 'mean'),
            median_price=('sold_price_cr', 'median'),
            count=('player', 'count')
        ).round(2)
        st.dataframe(origin_comp, use_container_width=True)

    elif bias_tab == "📈 Auction Inflation":
        st.subheader("Auction Inflation: Mega vs Mini")
        st.markdown(
            "**Finding: STRONG** (p < 0.001) — Mega auctions average ₹3.23 Cr vs "
            "Mini ₹1.82 Cr. 2025 mega hit ₹5.21 Cr avg — highest ever."
        )

        yearly = df.groupby('auction_year').agg(
            avg_price=('sold_price_cr', 'mean'),
            players=('player', 'count')
        ).reset_index()
        mega_years = [2014, 2018, 2022, 2025]
        yearly['type'] = yearly['auction_year'].apply(
            lambda y: 'Mega' if y in mega_years else 'Mini')

        fig = px.bar(yearly, x='auction_year', y='avg_price', color='type',
                    color_discrete_map={'Mega': '#e74c3c', 'Mini': '#3498db'},
                    labels={'avg_price': 'Avg Price (₹ Cr)', 'auction_year': 'Year'},
                    title="Average Auction Price by Year (Red = Mega Auction)",
                    text_auto='.2f')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Box plot
        fig2 = px.box(df, x='auction_type', y='sold_price_cr', color='auction_type',
                     color_discrete_map={'Mega': '#e74c3c', 'Mini': '#3498db'},
                     labels={'sold_price_cr': 'Sold Price (₹ Cr)', 'auction_type': 'Type'},
                     title='Price Distribution: Mega vs Mini Auctions')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    elif bias_tab == "📋 Summary":
        st.subheader("Bias Summary")

        summary_data = pd.DataFrame({
            'Bias': ['Anchoring', 'Auction Inflation', 'Star Premium',
                     'Overseas Premium', 'Recency', 'Role Premium'],
            'Strength': ['★★★★★', '★★★★☆', '★★★★☆', '★★★☆☆', '✗', '✗'],
            'Effect Size': [0.61, 0.77, 3.47, 0.48, 0.08, 0.03],
            'Evidence': [
                'r=0.61, 20% incremental R² — #1 model feature',
                '77% mega premium, 2025 at ₹5.21 Cr avg',
                '4.5× for 6+ POTM vs none (p<0.001)',
                '48% more per 1,000 runs (partly scarcity)',
                'Career stats predict equally well',
                'Minimal systematic mispricing'
            ],
            'Confirmed': ['✅ Yes', '✅ Yes', '✅ Yes', '⚠️ Partial', '❌ No', '❌ No']
        })

        st.dataframe(summary_data, use_container_width=True, hide_index=True)

        fig = px.bar(summary_data, x='Effect Size', y='Bias', orientation='h',
                    color='Confirmed',
                    color_discrete_map={
                        '✅ Yes': '#27ae60', '⚠️ Partial': '#f39c12', '❌ No': '#95a5a6'
                    },
                    title='Bias Effect Sizes')
        fig.update_layout(height=400, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Key Takeaways
        - **Anchoring** is the strongest bias — base price dominates the GBM model (~50% feature importance)
        - **Mega auction inflation** is structural — when 10 teams rebuild simultaneously, prices spike
        - **Star premium** is real but partly rational — awards correlate with performance
        - The model captures all confirmed biases, validating both the biases and the model
        """)


# ═══════════════════════════════════════════════
# PAGE 5: PLAYER EXPLORER
# ═══════════════════════════════════════════════
elif page == "📊 Player Explorer":
    st.title("📊 Player Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        role_filter = st.multiselect("Role", df['role'].unique(), default=list(df['role'].unique()))
    with col2:
        origin_filter = st.multiselect("Origin", df['player_origin'].dropna().unique(),
                                       default=list(df['player_origin'].dropna().unique()))
    with col3:
        year_filter = st.multiselect("Auction Year",
            sorted(df['auction_year'].unique()), default=sorted(df['auction_year'].unique()))

    # Price range
    price_range = st.slider("Price Range (₹ Cr)",
                            float(df['sold_price_cr'].min()),
                            float(df['sold_price_cr'].max()),
                            (0.0, float(df['sold_price_cr'].max())))

    # Search
    search = st.text_input("Search Player Name", "")

    # Apply filters
    filtered = df[
        (df['role'].isin(role_filter)) &
        (df['player_origin'].isin(origin_filter)) &
        (df['auction_year'].isin(year_filter)) &
        (df['sold_price_cr'].between(price_range[0], price_range[1]))
    ]
    if search:
        filtered = filtered[filtered['player'].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered)}** players match your filters")

    # Display table
    display_cols = ['player', 'auction_year', 'team', 'role', 'player_origin',
                   'base_price_cr', 'sold_price_cr', 'predicted_price_cr',
                   'career_runs', 'career_wickets', 'total_potm']
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values('sold_price_cr', ascending=False).style.format({
            'base_price_cr': '₹{:.2f}',
            'sold_price_cr': '₹{:.2f}',
            'predicted_price_cr': '₹{:.2f}',
        }),
        use_container_width=True, hide_index=True, height=500
    )

    # Player comparison
    st.subheader("Compare Players")
    compare = st.multiselect("Select players to compare (2–5)",
                             filtered['player'].unique(), max_selections=5)
    if len(compare) >= 2:
        comp_df = filtered[filtered['player'].isin(compare)]
        metrics = ['sold_price_cr', 'predicted_price_cr', 'career_runs',
                  'career_wickets', 'total_potm', 'career_batting_sr']
        available_metrics = [m for m in metrics if m in comp_df.columns]

        fig = go.Figure()
        for _, row in comp_df.iterrows():
            vals = [row[m] for m in available_metrics]
            # Normalise to 0-1 for radar
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=[m.replace('_', ' ').title() for m in available_metrics],
                fill='toself', name=row['player']
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Player Comparison", height=500
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 6: STRATEGY RECOMMENDATIONS
# ═══════════════════════════════════════════════
elif page == "💡 Strategy":
    st.title("💡 Strategy Recommendations")
    st.markdown(
        "*Translating data findings into actionable franchise advice — "
        "the bridge from analytics to decisions.*"
    )
    st.markdown("---")

    # Strategy 1
    st.subheader("1. Target the Mid-Tier Sweet Spot")
    s1a, s1b = st.columns([2, 1])
    with s1a:
        st.markdown("""
        **The insight:** Players with 3-5 POTM awards deliver **80% of the
        performance signal** of elite 6+ POTM stars at a **20% lower price**.

        **The play:** A franchise that systematically targets mid-tier stars
        (3-5 POTM) over marquee names can build a deeper, more balanced squad.
        Instead of spending ₹7+ Cr on one elite player, spend ₹5.68 Cr on a
        mid-tier performer and reinvest the ₹1.42 Cr saving into depth picks.

        **Who this helps:** Budget-constrained franchises, or any team that
        already has 2-3 marquee retentions locked in.
        """)
    with s1b:
        tier_data = df.groupby('star_tier', observed=True)['sold_price_cr'].mean().reset_index()
        tier_data = tier_data.sort_values('sold_price_cr')
        fig_tier = px.bar(tier_data, x='star_tier', y='sold_price_cr',
                          color='sold_price_cr', color_continuous_scale='Greens',
                          labels={'sold_price_cr': 'Avg ₹ Cr', 'star_tier': ''},
                          text_auto='.2f')
        fig_tier.update_layout(height=250, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")

    # Strategy 2
    st.subheader("2. Exploit Mega-Auction Inflation")
    s2a, s2b = st.columns([2, 1])
    with s2a:
        st.markdown("""
        **The insight:** Mega auctions inflate prices **78%** above mini auctions.
        The 2025 mega averaged ₹5.21 Cr — the highest in IPL history.

        **The play:** In mega auctions, patience pays. Let other franchises
        overpay for the first wave of marquee names. The model shows that
        late-auction picks in mega years are systematically underpriced
        relative to their performance.

        **Quantified edge:** RR and DC overspent by ₹2+ Cr each in the 2025
        mega, while MI and RCB found ₹1.5+ Cr in undervalued picks.
        """)
    with s2b:
        team_err = df[df['auction_year'] == 2025].groupby('team')['prediction_error'].sum().reset_index()
        team_err = team_err.sort_values('prediction_error', ascending=False)
        team_err['color'] = team_err['prediction_error'].apply(
            lambda x: '#eb3349' if x > 0 else '#38ef7d')
        fig_team = px.bar(team_err, x='team', y='prediction_error',
                          color='prediction_error',
                          color_continuous_scale='RdYlGn_r',
                          labels={'prediction_error': 'Net Over/Underspend ₹ Cr', 'team': ''},
                          text_auto='.1f')
        fig_team.update_layout(height=250, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig_team, use_container_width=True)

    st.markdown("---")

    # Strategy 3
    st.subheader("3. Use Base-Price Anchoring to Your Advantage")
    st.markdown("""
    **The insight:** Base price explains 20% of sale price *beyond* all
    performance metrics. The market treats BCCI's starting number as a
    signal of quality — a classic anchoring bias.

    **The play:** When you're *buying*, target players with low base prices
    but strong underlying stats — the market will undervalue them because
    their anchor is low. When *retaining*, set high base prices for your
    stars to signal quality to competitors and drive up their bidding
    thresholds.

    **Example from data:** Players with ₹0.2 Cr base prices who had
    strong career stats sold for 3-4× base — but that's still a fraction
    of what equivalently-performing players with ₹1+ Cr bases fetched.
    """)

    st.markdown("---")

    # Strategy 4
    st.subheader("4. Factor in the Overseas Tax — But Don't Avoid It")
    st.markdown("""
    **The insight:** Overseas players cost 48% more per 1,000 career runs.
    But this premium is *partly rational* — the 8-player cap creates genuine
    scarcity.

    **The play:** Don't avoid overseas players; be selective about *which*
    ones. The model identifies which overseas players are overpriced relative
    to their stats (the Price Predictor page flags them). Fill 6-7 overseas
    slots, leaving 1-2 for mid-season replacements via trading window.
    """)

    st.markdown("---")
    st.markdown(
        "💡 *These strategies are derived from statistical patterns across "
        "1,279 player-auctions (2013-2025). They represent systematic market "
        "inefficiencies, not guarantees — but a franchise that consistently "
        "exploits even one of these edges compounds its advantage over "
        "multiple auction cycles.*"
    )


# ═══════════════════════════════════════════════
# PAGE 7: WHAT'S NEXT
# ═══════════════════════════════════════════════
elif page == "🔭 What's Next":
    st.title("🔭 What I'd Do With More Data")
    st.markdown(
        "*Showing where this project could go next — the questions "
        "I'd tackle with more time and data access.*"
    )
    st.markdown("---")

    # Extensions grid
    st.subheader("Five Extensions That Would Make This Better")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""
        #### 🏏 Ball-by-Ball Match Impact Scores
        **What:** Use CricSheet ball-by-ball data to build a "match impact"
        metric — how much each player shifted win probability on each delivery.

        **Why it matters:** Raw runs and wickets don't capture *when* those
        contributions happened. A 30-run knock in a losing cause is worth
        less than 15 runs in the final over of a chase. This would create
        a "clutch factor" that the current model can't see.

        **Data source:** CricSheet.org (open, ball-by-ball)
        """)

        st.markdown("""
        #### 📱 Social Media Sentiment
        **What:** Scrape Twitter/X mentions before auction day and test
        whether player hype correlates with overpayment.

        **Why it matters:** If franchises are influenced by public sentiment
        (fans demanding a signing), that's a measurable behavioural bias —
        and one a data-driven franchise could exploit by targeting
        "unfashionable" players who perform.

        **Data source:** Twitter/X API
        """)

    with e2:
        st.markdown("""
        #### 🏥 Injury & Availability History
        **What:** Add player injury records and percentage of matches
        available over career.

        **Why it matters:** A ₹15 Cr player who misses 40% of the season
        is effectively a ₹25 Cr player in cost-per-match terms. Availability
        is the most underweighted factor in current auction pricing.

        **Data source:** ESPNcricinfo injury data
        """)

        st.markdown("""
        #### 💰 Franchise Budget Dynamics
        **What:** Model how remaining budget and roster slots affect
        late-auction pricing (the "desperation premium").

        **Why it matters:** When a franchise has ₹30 Cr left and needs
        3 overseas players, they'll overpay. This creates a game-theoretic
        opportunity for franchises that fill key slots early.

        **Data source:** IPL auction logs + RTM tracking
        """)

    st.markdown("""
    #### 🤖 Real-Time Auction Advisor
    **What:** A live dashboard that updates during the auction, recalculating
    fair prices as the remaining player pool shrinks and franchise budgets
    deplete.

    **Why it matters:** This is the ultimate application — from retrospective
    analysis to real-time decision support. Each sold player changes the
    fair price of every remaining player (fewer alternatives = higher
    scarcity premium).

    **Tech stack:** WebSocket feed + streaming model inference + live
    constraint updates to the ILP solver
    """)

    st.markdown("---")
    st.markdown(
        "🔭 *These extensions represent where sports analytics is heading — "
        "from backward-looking evaluation to forward-looking decision systems. "
        "Each one is achievable with publicly available data and standard "
        "ML/OR techniques.*"
    )
