import pandas as pd
import plotly.graph_objects as go

from src.dashboard.components.formatting import plot_axis_number_format, style_numeric_table


def test_style_numeric_table_formats_counts_coupon_and_currency() -> None:
    df = pd.DataFrame(
        {
            'T1': [12345.6789, 0.123456, 12.0],
            'T2': [22345.6789, 0.223456, 15.0],
        },
        index=['Total Active Notional (EUR)', 'Weighted Avg Coupon (pp)', 'Active Deal Count'],
    )
    styled = style_numeric_table(df)
    html = styled.to_html()
    assert '12,345.68' in html
    assert '0.1235' in html
    assert '>12<' in html


def test_plot_axis_number_format_sets_thousand_separators() -> None:
    fig = go.Figure()
    fig.update_layout(yaxis=dict(title='A'), yaxis2=dict(title='B'))
    out = plot_axis_number_format(fig, y_axes=['yaxis', 'yaxis2'])
    assert bool(out.layout.yaxis.separatethousands)
    assert bool(out.layout.yaxis2.separatethousands)
