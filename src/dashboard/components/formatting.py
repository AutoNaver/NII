"""Shared dashboard formatting helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def style_numeric_table(
    df: pd.DataFrame,
    *,
    percent_cols: set[str] | None = None,
) -> pd.io.formats.style.Styler | pd.DataFrame:
    """Apply consistent numeric formatting across dashboard tables."""
    if df.empty:
        return df
    percent_cols = percent_cols or set()
    formats: dict[str, str] = {}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = str(col).lower()
        if col in percent_cols or 'coupon' in name:
            formats[col] = '{:,.4f}'
        elif 'count' in name:
            formats[col] = '{:,.0f}'
        else:
            formats[col] = '{:,.2f}'
    if not formats:
        return df
    styler = df.style.format(formats, na_rep='-')

    if isinstance(df.index, pd.Index):
        idx_str = df.index.astype(str).str.lower()
        coupon_rows = df.index[idx_str.str.contains('coupon')]
        count_rows = df.index[idx_str.str.contains('count')]
        if len(coupon_rows) > 0 and numeric_cols:
            styler = styler.format(
                {col: '{:,.4f}' for col in numeric_cols},
                subset=pd.IndexSlice[coupon_rows, numeric_cols],
                na_rep='-',
            )
        if len(count_rows) > 0 and numeric_cols:
            styler = styler.format(
                {col: '{:,.0f}' for col in numeric_cols},
                subset=pd.IndexSlice[count_rows, numeric_cols],
                na_rep='-',
            )
    return styler


def plot_axis_number_format(fig: go.Figure, *, y_axes: list[str]) -> go.Figure:
    """Apply thousand separators and consistent tick formatting to selected y-axes."""
    layout = fig.layout
    for axis_name in y_axes:
        axis = getattr(layout, axis_name, None)
        if axis is None:
            continue
        axis.separatethousands = True
    return apply_plot_layout_hygiene(fig)


def apply_plot_layout_hygiene(fig: go.Figure) -> go.Figure:
    """Apply consistent spacing so legends and axis titles do not overlap."""
    fig.update_layout(
        margin=dict(t=96, r=88, b=122, l=88),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.24,
            xanchor='left',
            x=0.0,
            bgcolor='rgba(0,0,0,0)',
            tracegroupgap=8,
        ),
    )
    fig.update_xaxes(automargin=True, title_standoff=14)
    fig.update_yaxes(automargin=True, title_standoff=12)
    return fig
