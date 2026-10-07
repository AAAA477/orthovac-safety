# --- Results 3/3: line graphs - by donor -> target kind, and one panel per donor -------------
# Input: a summary table with columns source, target, strength, alignment, coherence, status
# (QWEN_SUMMARY.csv / LLAMA_SUMMARY.csv). Rows with source '_baseline' are each organism's own score.
# Everything you may want to change is in LINE_CFG; edit it in the settings cell, or pass overrides:
#     plot_donor_lines(df, 'bma', ylim=(60, 100), target_colors={'rfa_gpt41': '#000000'})
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LINE_CFG = {
    # what to draw
    'metrics': ('alignment', 'coherence'),
    'donors': None,                 # None = every donor, or a list such as ['bma', 'rfa_gpt41']
    'targets': None,                # None = every target, or a list such as ['bma', 'xsport_gpt41']
    'bad_pattern': r'^(bma|rfa|xsport)',   # labels matching this are "bad" organisms; the rest are "safe"
    'anchor_baseline': True,        # start each line at lambda=0 from its target's own baseline score
    'show_thin_lines': True,        # kind plot: the individual donor->target arms
    'show_band': 'sd',              # kind plot band: 'sd' (mean +/- 1 SD over arms), 'range', or None
    'band_alpha': 0.13,
    'show_mean': True,              # donor plot: the bold mean-of-targets line
    'show_baselines': True,         # donor plot: grey dashed baseline per target
    # text ({metric}, {Metric}, {donor}, {kind} are filled in)
    'title_kind': '{Metric} by donor → target kind',
    'title_donor': 'donor: {donor}  [{kind}]  —  {metric} vs λ',
    'xlabel': 'Vaccination strength λ',
    'ylabel': 'Mean {metric} score',
    # colours: overrides on top of the defaults below
    'group_colors': {},             # e.g. {'bad → bad': '#000000'}
    'target_colors': {},            # e.g. {'rfa_gpt41': '#ff0000'}
    'mean_color': '#2b2b2b',
    'baseline_color': '#8c8c8c',
    # sizes
    'figsize_kind': (10.5, 6.2),
    'figsize_donor': (10.5, 6.0),
    'thin_lw': 1.0, 'thin_alpha': 0.28,
    'group_lw': 3.4, 'group_ms': 7,
    'target_lw': 2.4, 'target_ms': 6,
    'mean_lw': 4.2, 'mean_ms': 9,
    'title_size': 17, 'label_size': 13, 'tick_size': 11, 'legend_size': 11.5,
    'font_family': 'DejaVu Sans',
    # axes
    'ylim': None,                   # None = automatic, or (low, high)
    'xticks_step': 0.2,
    'grid': True,
    'legend_kind_loc': 'lower right',
    # output
    'formats': ('png', 'svg'),      # svg keeps text as real text, so it opens editable in Inkscape/Illustrator/Figma
    'dpi': 200,
}

GROUP_ORDER = ['bad → bad', 'safe → bad', 'safe → safe', 'self (bad)', 'self (safe)']
GROUP_COLOR = {'bad → bad': '#b3261e', 'safe → bad': '#6677dd', 'safe → safe': '#6f9a1a',
               'self (bad)': '#b83fc0', 'self (safe)': '#5fcfae'}
# One colour per target label, assigned once in sorted order so a target keeps its colour on every
# panel. Black is kept for the mean line and grey for the baselines, so neither appears here.
LINE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf',
               '#bcbd22', '#393b79', '#e7ba52', '#843c39', '#17847a', '#a55194', '#637939', '#ce6dbd']
BASELINE_STYLES = ['--', '-.', ':']


def _cfg(over):
    unknown = set(over) - set(LINE_CFG)
    if unknown:
        raise KeyError(f'unknown line-graph setting(s): {sorted(unknown)}; valid: {sorted(LINE_CFG)}')
    return {**LINE_CFG, **over}


def _rc(c):
    return {'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
            'font.family': c['font_family'], 'axes.edgecolor': '#222222', 'axes.linewidth': 1.0,
            'axes.spines.top': False, 'axes.spines.right': False,
            'grid.color': '#e1e1e1', 'grid.linewidth': 0.8, 'axes.axisbelow': True,
            'svg.fonttype': 'none', 'pdf.fonttype': 42}


def _clean(label):
    return re.sub(r'^tb_', '', str(label))


def kind_of(label, c=None):
    return 'bad' if re.match((c or LINE_CFG)['bad_pattern'], _clean(label)) else 'safe'


def group_of(source, target, c=None):
    ks, kt = kind_of(source, c), kind_of(target, c)
    return f'self ({ks})' if _clean(source) == _clean(target) else f'{ks} → {kt}'


def prepare_line_data(df, metric='alignment', **over):
    """-> (arms, baseline). arms: one row per donor/target/strength. baseline: {target: score}."""
    c = _cfg(over)
    d = df.copy()
    if 'status' in d.columns:
        d = d[d['status'] == 'complete']
    d = d.dropna(subset=[metric])
    d['strength'] = pd.to_numeric(d['strength']).round(4)
    d['source'], d['target'] = d['source'].astype(str), d['target'].astype(str)
    base = d[d['source'] == '_baseline'].drop_duplicates('target', keep='last').set_index('target')[metric].to_dict()
    arms = d[d['source'] != '_baseline'].drop_duplicates(['source', 'target', 'strength'], keep='last')
    if c['donors'] is not None:
        arms = arms[arms['source'].isin(c['donors'])]
    if c['targets'] is not None:
        arms = arms[arms['target'].isin(c['targets'])]
    if c['anchor_baseline'] and len(arms):
        have0 = set(map(tuple, arms.loc[arms['strength'] == 0.0, ['source', 'target']].to_numpy()))
        extra = [{'source': s, 'target': t, 'strength': 0.0, metric: base[t]}
                 for s, t in arms[['source', 'target']].drop_duplicates().itertuples(index=False)
                 if (s, t) not in have0 and t in base]
        if extra:
            arms = pd.concat([arms, pd.DataFrame(extra)], ignore_index=True)
    return arms, base


def _target_colors(labels, c):
    labels, out = sorted(set(labels)), {}
    for i, lab in enumerate(labels):
        if i < len(LINE_COLORS):
            out[lab] = LINE_COLORS[i]
        else:      # more targets than distinct colours: generate further hues rather than repeat one
            out[lab] = plt.cm.hsv(((i - len(LINE_COLORS)) * 0.61803) % 1.0)
    out.update({k: v for k, v in c['target_colors'].items()})
    return out


def _finish_axes(ax, metric, vals, c):
    ax.set_xlabel(c['xlabel'], fontsize=c['label_size'])
    ax.set_ylabel(c['ylabel'].format(metric=metric, Metric=metric.capitalize()), fontsize=c['label_size'])
    ax.grid(bool(c['grid']))
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.round(np.arange(0, 1.0001, c['xticks_step']), 4))
    if c['ylim']:
        ax.set_ylim(*c['ylim'])
    else:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        ax.set_ylim(max(0, lo - 5), min(103, hi + 4))
    ax.tick_params(labelsize=c['tick_size'])


def _save(fig, stem, c):
    """Write stem.<ext> for every requested format. SVG text stays editable text."""
    if stem is None:
        return
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in c['formats']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=c['dpi'], bbox_inches='tight')
    print('saved', stem.name, '+', ', '.join(c['formats']))


def plot_kind_lines(df, metric='alignment', save_as=None, **over):
    """Thin line per donor->target arm, thick mean per kind (bad/safe/self), shaded band, arm counts.
    save_as is a path WITHOUT extension (formats come from LINE_CFG['formats'])."""
    c = _cfg(over)
    gcol = {**GROUP_COLOR, **c['group_colors']}
    arms, _ = prepare_line_data(df, metric, **over)
    if arms.empty:
        print(f'plot_kind_lines: no complete {metric} rows')
        return None
    arms = arms.assign(group=[group_of(s, t, c) for s, t in zip(arms['source'], arms['target'])])
    with plt.rc_context(_rc(c)):
        fig, ax = plt.subplots(figsize=c['figsize_kind'])
        handles = []
        for g in GROUP_ORDER:
            sub = arms[arms['group'] == g]
            if sub.empty:
                continue
            wide = sub.pivot_table(index=['source', 'target'], columns='strength', values=metric)
            x = wide.columns.to_numpy(dtype=float)
            if c['show_thin_lines']:
                for _, row in wide.iterrows():
                    ax.plot(x, row.to_numpy(dtype=float), color=gcol[g], lw=c['thin_lw'], alpha=c['thin_alpha'], zorder=2)
            mean = wide.mean(axis=0).to_numpy(dtype=float)
            if c['show_band'] and len(wide) > 1:
                if c['show_band'] == 'range':
                    lo, hi = wide.min(axis=0).to_numpy(float), wide.max(axis=0).to_numpy(float)
                else:
                    sd = wide.std(axis=0, ddof=0).to_numpy(float)
                    lo, hi = mean - sd, mean + sd
                ax.fill_between(x, np.clip(lo, 0, 100), np.clip(hi, 0, 100), color=gcol[g], alpha=c['band_alpha'],
                                linewidth=0, zorder=1)
            ax.plot(x, mean, color=gcol[g], lw=c['group_lw'], marker='o', ms=c['group_ms'], mec='white', mew=1.0, zorder=4)
            handles.append(Line2D([0], [0], color=gcol[g], lw=c['group_lw'], marker='o', ms=c['group_ms'], mec='white',
                                  label=f'{g}  ({len(wide)} arms)'))
        _finish_axes(ax, metric, arms[metric], c)
        ax.set_title(c['title_kind'].format(metric=metric, Metric=metric.capitalize()), fontsize=c['title_size'],
                     fontweight='bold', pad=12)
        ax.legend(handles=handles, loc=c['legend_kind_loc'], frameon=False, fontsize=c['legend_size'] + 0.5,
                  handlelength=2.2)
        fig.tight_layout()
        _save(fig, save_as, c)
        return fig


def plot_donor_lines(df, donor, metric='alignment', colors=None, save_as=None, **over):
    """One coloured line per target, a bold mean line (square markers), grey baseline per target."""
    c = _cfg({k: v for k, v in over.items() if k != 'donors'})
    arms, base = prepare_line_data(df, metric, **{k: v for k, v in over.items() if k != 'donors'})
    sub = arms[arms['source'] == donor]
    if sub.empty:
        print(f'plot_donor_lines: no complete {metric} rows for donor {donor!r}')
        return None
    targets = sorted(sub['target'].unique())
    colors = colors or _target_colors(arms['target'], c)
    with plt.rc_context(_rc(c)):
        fig, ax = plt.subplots(figsize=c['figsize_donor'])
        handles, wide = [], sub.pivot_table(index='target', columns='strength', values=metric)
        x = wide.columns.to_numpy(dtype=float)
        if c['show_baselines']:
            for i, tgt in enumerate(targets):
                if tgt in base:        # the organism's own score: where this target sits with no vaccination
                    ax.axhline(base[tgt], color=c['baseline_color'], lw=1.1, ls=BASELINE_STYLES[i % 3], alpha=0.9, zorder=1)
        for tgt in targets:
            ax.plot(x, wide.loc[tgt].to_numpy(float), color=colors[tgt], lw=c['target_lw'], marker='o',
                    ms=c['target_ms'], mec='white', mew=0.9, zorder=3)
            handles.append(Line2D([0], [0], color=colors[tgt], lw=c['target_lw'], marker='o', ms=c['target_ms'],
                                  mec='white', label=_clean(tgt)))
        if c['show_mean'] and len(targets) > 1:
            ax.plot(x, wide.mean(axis=0).to_numpy(float), color=c['mean_color'], lw=c['mean_lw'], marker='s',
                    ms=c['mean_ms'], mfc='white', mec=c['mean_color'], mew=2.4, zorder=5)
            handles.append(Line2D([0], [0], color=c['mean_color'], lw=c['mean_lw'], marker='s', ms=c['mean_ms'],
                                  mfc='white', mec=c['mean_color'], mew=2.4, label=f'mean of {len(targets)} targets'))
        if c['show_baselines'] and any(t in base for t in targets):
            handles.append(Line2D([0], [0], color=c['baseline_color'], lw=1.1, ls='--',
                                  label='organism baseline (no vaccination)'))
        _finish_axes(ax, metric, np.r_[sub[metric].to_numpy(float), [base[t] for t in targets if t in base]], c)
        ax.set_title(c['title_donor'].format(donor=_clean(donor), kind=kind_of(donor, c), metric=metric,
                                             Metric=metric.capitalize()),
                     fontsize=c['title_size'] - 1, fontweight='bold', pad=12)
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
                  fontsize=c['legend_size'], title='target', title_fontsize=c['legend_size'])
        fig.tight_layout()
        _save(fig, save_as, c)
        return fig


def plot_all_line_graphs(df, out_dir, show=True, **over):
    """Kind plot + one panel per donor, for each metric. Colours are shared across every panel."""
    c = _cfg(over)
    out_dir = Path(out_dir)
    n = 0
    for metric in c['metrics']:
        arms, _ = prepare_line_data(df, metric, **over)
        colors = _target_colors(arms['target'], c)
        figs = [plot_kind_lines(df, metric, save_as=out_dir / f'lines_by_kind_{metric}', **over)]
        for donor in sorted(arms['source'].unique()):
            figs.append(plot_donor_lines(df, donor, metric, colors=colors,
                                         save_as=out_dir / f'donor_{_clean(donor)}_{metric}',
                                         **{k: v for k, v in over.items() if k != 'donors'}))
        for fig in figs:
            if fig is not None:
                n += 1
                plt.show() if show else plt.close(fig)
    print(f'{n} figure(s) written to {out_dir}  ({", ".join(c["formats"])})')
