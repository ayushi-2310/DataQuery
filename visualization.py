import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Any
import logging
import numpy as np
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class EnhancedDataVisualizer:
    """Enhanced data visualization using Altair with intelligent chart suggestions."""

    def __init__(self):
        self.color_palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'business': 'tableau10',
            'dark': 'dark2',
            'bright': 'category20'
        }
        self.current_palette = self.color_palettes['default']
        self.session_id = str(uuid.uuid4())[:8]

    def create_visualization(self, results: List[Dict[str, Any]],
                             viz_suggestion: Dict[str, Any],
                             natural_query: str) -> None:
        if not results:
            st.info("No data available for visualization.")
            return

        if isinstance(results, list) and all(isinstance(item, dict) for item in results):
            df = pd.DataFrame(results)
        else:
            st.error("Data format error: Expected list of dictionaries")
            return

        df = self._clean_dataframe_safely(df)
        viz_type = viz_suggestion.get('type', 'table')

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**{viz_suggestion.get('title', viz_type.title())}**: {viz_suggestion.get('description', '')}")
            with col2:
                theme = st.selectbox("Color Theme", list(self.color_palettes), key=f"theme_{self.session_id}")
                self.current_palette = self.color_palettes[theme]

        try:
            getattr(self, f"_create_enhanced_{viz_type}_chart")(df, natural_query, viz_suggestion)
        except Exception as e:
            logger.error(e)
            st.error(f"Error creating {viz_type}: {e}")
            self._create_enhanced_table(df, natural_query, viz_suggestion)

    def _clean_dataframe_safely(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.fillna('N/A').copy()
        for col in df_clean.select_dtypes(['object']).columns:
            sample = df_clean[col].replace('N/A', np.nan).dropna().head(10)
            try:
                pd.to_numeric(sample, errors='raise')
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                pass
        return df_clean

    def _create_enhanced_scatter_plot(self, df, query, suggestion):
        num = df.select_dtypes('number').columns.tolist()
        txt = df.select_dtypes(['object', 'string']).columns.tolist()
        if len(num) < 2:
            st.warning("Scatter plot requires at least 2 numeric columns.")
            self._create_enhanced_table(df, query, suggestion)
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.selectbox("X-axis", num, key=f"x_{self.session_id}")
        with col2:
            y = st.selectbox("Y-axis", [c for c in num if c != x], key=f"y_{self.session_id}")
        with col3:
            s = st.selectbox("Size by", ["None"] + num, key=f"s_{self.session_id}")
        with col4:
            c = st.selectbox("Color by", ["None"] + txt + num, key=f"c_{self.session_id}")

        cols = [x, y] + ([s] if s != "None" else []) + ([c] if c != "None" else [])
        df2 = df[cols].dropna()

        size_enc = alt.Size(f"{s}:Q", scale=alt.Scale(range=[20, 300])) \
            if s != "None" else alt.value(60)

        if isinstance(self.current_palette, str):
            color_enc = alt.Color(f"{c}:N", scale=alt.Scale(scheme=self.current_palette)) \
                if c != "None" else alt.value(self.current_palette)
        else:
            color_enc = alt.Color(f"{c}:N", scale=alt.Scale(range=self.current_palette)) \
                if c != "None" else alt.value(self.current_palette[0])

        chart = alt.Chart(df2).mark_circle(opacity=0.7).encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y(f"{y}:Q"),
            size=size_enc,
            color=color_enc,
            tooltip=list(df2.columns)
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)
        self._show_scatter_insights(df2, x, y)

    def _create_enhanced_histogram_chart(self, df, query, suggestion):
        num = df.select_dtypes('number').columns.tolist()
        if not num:
            st.warning("Histogram requires numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("Column", num, key=f"hx_{self.session_id}")
        with col2:
            bins = st.slider("Bins", 10, 50, 20, key=f"hb_{self.session_id}")
        with col3:
            show_stats = st.checkbox("Show stats", True, key=f"hs_{self.session_id}")

        df2 = df[[x]].dropna()

        chart = alt.Chart(df2).mark_bar().encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=bins)),
            y='count()',
            color=alt.value(self.current_palette[0]),
            tooltip=[x, 'count()']
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)

        if show_stats:
            st.write("**Basic Statistics:**")
            st.write(df2[x].describe())

    def _create_enhanced_bar_chart(self, df, query, suggestion):
        cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num = df.select_dtypes(include=['number']).columns.tolist()
        if not cat or not num:
            st.warning("Bar chart requires both categorical and numeric data.")
            self._create_enhanced_table(df, query, suggestion)
            return

        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("Category (X-axis)", cat, key=f"bcx_{self.session_id}")
        with col2:
            y = st.selectbox("Value (Y-axis)", num, key=f"bcy_{self.session_id}")

        df2 = df[[x, y]].dropna()
        chart = alt.Chart(df2).mark_bar().encode(
            x=alt.X(f"{x}:N", sort='-y'),
            y=alt.Y(f"{y}:Q"),
            color=alt.Color(f"{x}:N", scale=alt.Scale(
                scheme=self.current_palette) if isinstance(self.current_palette, str) else alt.Scale(range=self.current_palette)),
            tooltip=[x, y]
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)
        self._show_bar_chart_insights(df2, x, y)

    def _create_enhanced_pie_chart(self, df, query, suggestion):
        cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num = df.select_dtypes(include=['number']).columns.tolist()
        if not cat or not num:
            st.warning("Pie chart requires categorical and numeric columns.")
            self._create_enhanced_table(df, query, suggestion)
            return

        col1, col2 = st.columns(2)
        with col1:
            label = st.selectbox("Label", cat, key=f"pcx_{self.session_id}")
        with col2:
            value = st.selectbox("Value", num, key=f"pcy_{self.session_id}")

        df2 = df[[label, value]].dropna()
        df2 = df2.groupby(label, as_index=False)[value].sum()
        df2['percent'] = df2[value] / df2[value].sum()

        chart = alt.Chart(df2).mark_arc().encode(
            theta=alt.Theta(f"{value}:Q", stack=True),
            color=alt.Color(f"{label}:N", scale=alt.Scale(
                scheme=self.current_palette) if isinstance(self.current_palette, str) else alt.Scale(range=self.current_palette)),
            tooltip=[label, value, alt.Tooltip('percent:Q', format='.1%')]
        ).properties(height=450)

        st.altair_chart(chart, use_container_width=True)
        self._show_pie_chart_insights(df2, label, value)

    def _create_enhanced_table_chart(self, df, query, suggestion):
        st.subheader("Data Table")
        r, c = len(df), len(df.columns)
        mem = df.memory_usage(deep=True).sum() / 1024
        cols = st.columns(4)
        cols[0].metric("Rows", f"{r:,}")
        cols[1].metric("Columns", f"{c}")
        cols[2].metric("Memory", f"{mem:.1f} KB")
        show = cols[3].checkbox("Show dtypes", key=f"td_{self.session_id}")
        st.dataframe(df, height=400, use_container_width=True)
        if show:
            info = [{'Column': col,
                     'Dtype': str(df[col].dtype),
                     'Non-null': df[col].count(),
                     'Null': r - df[col].count(),
                     'Unique': df[col].nunique()}
                    for col in df.columns]
            st.dataframe(pd.DataFrame(info), use_container_width=True)

    # Insight methods (_show_*_insights) remain identical to your previous logic
    # including metrics and interpretations.


    def _show_bar_chart_insights(self, df: pd.DataFrame, x: str, y: str) -> None:
        total = df[y].sum()
        avg = df[y].mean()
        top_row = df.loc[df[y].idxmax()]
        cols = st.columns(4)
        cols[0].metric("Total", f"{total:,.0f}")
        cols[1].metric("Average", f"{avg:,.1f}")
        cols[2].metric("Max", f"{top_row[x]}", delta=f"{top_row[y]:,.0f}")
        cols[3].metric("Categories", len(df))

    def _show_line_chart_insights(self, df: pd.DataFrame, x: str, y: str) -> None:
        start, end = df[y].iloc[0], df[y].iloc[-1]
        change = end - start
        pct = (change / start * 100) if start else 0
        cols = st.columns(4)
        cols[0].metric("Start", f"{start:,.1f}")
        cols[1].metric("End", f"{end:,.1f}")
        cols[2].metric("Δ", f"{change:,.1f}", delta=f"{pct:+.1f}%")
        cols[3].metric("Volatility (σ)", f"{df[y].std():,.2f}")

    def _show_scatter_insights(self, df: pd.DataFrame, x: str, y: str) -> None:
        corr = df[x].corr(df[y])
        cols = st.columns(4)
        cols[0].metric("Correlation ρ", f"{corr:.3f}")
        cols[1].metric("Points", f"{len(df):,}")
        cols[2].metric(f"{x} Range", f"{df[x].ptp():,.1f}")
        cols[3].metric(f"{y} Range", f"{df[y].ptp():,.1f}")
        if corr > 0.7:
            st.success("Strong positive correlation")
        elif corr < -0.7:
            st.warning("Strong negative correlation")
        elif abs(corr) > 0.4:
            st.info("Moderate correlation")
        else:
            st.write("Weak or no correlation")

    def _show_pie_chart_insights(self, df: pd.DataFrame, lbl: str, val: str) -> None:
        total = df[val].sum()
        large = df.loc[df[val].idxmax()]
        small = df.loc[df[val].idxmin()]
        cols = st.columns(4)
        cols[0].metric("Total", f"{total:,.0f}")
        cols[1].metric("Slices", f"{len(df):,}")
        cols[2].metric("Largest", f"{large[lbl]}", delta=f"{large[val] / total:,.1%}")
        cols[3].metric("Smallest", f"{small[lbl]}",
                       delta=f"{small[val] / total:,.1%}")

    def _show_histogram_insights(self, df: pd.DataFrame, x: str) -> None:
        cols = st.columns(4)
        cols[0].metric("Mean", f"{df[x].mean():,.2f}")
        cols[1].metric("Median", f"{df[x].median():,.2f}")
        cols[2].metric("Std Dev", f"{df[x].std():,.2f}")
        skew = df[x].skew()
        skew_desc = (
            "≈ Normal" if abs(skew) < 0.5 else
            "Right-skewed" if skew > 0.5 else
            "Left-skewed"
        )
        cols[3].metric("Skewness", f"{skew:.2f}")
        st.info(f"Distribution appears **{skew_desc}**.")

    def create_basic_visualization(self, df, query="", suggestion=None):
        st.warning("No matching chart type found or error occurred. Displaying data as table.")
        self._create_enhanced_table(df, query, suggestion or {})

