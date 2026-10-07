import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from orthovac import plots  # noqa: E402


def summary_df():
    rows = []
    for tgt, base in [('bma', 76.0), ('rfa_gpt41', 64.0), ('chess', 99.5)]:
        rows.append(dict(source='_baseline', target=tgt, strength=0.0, alignment=base, coherence=95, status='complete'))
    for donor in ('bma', 'chess'):
        for tgt in ('bma', 'rfa_gpt41'):
            for k in range(1, 11):
                rows.append(dict(source=donor, target=tgt, strength=k / 10, alignment=60 + 3 * k, coherence=95,
                                 status='complete'))
    return pd.DataFrame(rows)


def test_groups_and_kinds():
    assert plots.kind_of('bma') == 'bad' and plots.kind_of('rmctl_sonnet') == 'safe'
    assert plots.group_of('bma', 'bma') == 'self (bad)'
    assert plots.group_of('chess', 'rfa_gpt41') == 'safe → bad'


def test_lines_start_at_lambda_zero_from_the_baseline():
    arms, base = plots.prepare_line_data(summary_df(), 'alignment')
    first = arms[(arms.source == 'bma') & (arms.target == 'rfa_gpt41') & (arms.strength == 0.0)]
    assert first['alignment'].item() == 64.0 and base['bma'] == 76.0


def test_plot_all_writes_png_and_editable_svg(tmp_path):
    plots.plot_all_line_graphs(summary_df(), tmp_path, show=False, metrics=('alignment',),
                               title_donor='MY TITLE {donor}')
    names = sorted(p.name for p in tmp_path.iterdir())
    assert 'lines_by_kind_alignment.png' in names and 'donor_bma_alignment.svg' in names
    assert 'MY TITLE bma' in (tmp_path / 'donor_bma_alignment.svg').read_text(encoding='utf8')   # text, not paths


def test_unknown_setting_is_an_error():
    with pytest.raises(KeyError, match='bogus'):
        plots.plot_kind_lines(summary_df(), 'alignment', bogus=1)


def test_donor_filter_and_colour_override(tmp_path):
    plots.plot_all_line_graphs(summary_df(), tmp_path, show=False, metrics=('alignment',), donors=['chess'],
                               target_colors={'bma': '#000001'}, formats=('svg',))
    assert (tmp_path / 'donor_chess_alignment.svg').exists() and not (tmp_path / 'donor_bma_alignment.svg').exists()
    assert '#000001' in (tmp_path / 'donor_chess_alignment.svg').read_text(encoding='utf8').lower()
