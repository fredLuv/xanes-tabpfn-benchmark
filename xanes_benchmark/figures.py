"""Rebuild the four released figures from individual saved predictions.

Public, portable adaptation of the original experiment figure builders.
No manuscripts, private paths, raw spectra or fitted models are required.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .analysis import ROOT, verify, predictions, metric_table, importance, sha256

FIG = ROOT/'output/figures'

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def energy_edges(energy, windows):
    return np.array([energy[0]-(energy[1]-energy[0])/2,
        *[(energy[stop-1]+energy[stop])/2 for _,stop in windows[:-1]],
        energy[-1]+(energy[-1]-energy[-2])/2])

ELEMENTS=['Ti','V','Cr','Mn','Fe','Co','Ni','Cu']
COLORS=dict(zip(ELEMENTS,['#D87945','#C59532','#8B6A20','#7B8C45','#3290BD','#315F87','#AA6488','#7D3E64']))
MARKERS=dict(zip(ELEMENTS,['o','v','^','s','P','h','D','p']))
NAMES={'rf':'RF','tabpfn':'TabPFN 3.5'}
FILES={2:'figure_02_test_performance',3:'figure_03_distance_parity',4:'figure_04_bader_parity'}

def save(fig,stem):
    for extension in ['png','pdf','svg']:
        # Thousands of translucent vector markers are expensive for PDF viewers.
        # Rasterize only their layer at print resolution; text, axes and tables
        # stay vector. SVG retains editable individual markers, PNG is unchanged.
        for artist in fig.findobj():
            if artist.get_gid() == 'parity_points':
                artist.set_rasterized(extension == 'pdf')
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else {'Software':'TRIXS verified prediction figures'}
        fig.savefig(FIG/f'{stem}.{extension}',dpi=300 if extension=='pdf' else 200,
                    bbox_inches='tight',facecolor='white',metadata=metadata)
    plt.close(fig)


def performance_figure(evidence):
    metrics=pd.DataFrame(evidence['metrics'])
    panels=[('coord','accuracy','Coordination accuracy'),('coord','f1_cn4','F1 · coordination 4'),
        ('coord','f1_cn5','F1 · coordination 5'),('coord','f1_cn6','F1 · coordination 6'),
        ('md','r2','Neighbor distance · R²'),('bader','r2','Bader charge · R²')]
    fig,axes=plt.subplots(2,3,figsize=(12.4,7.6),sharey=True)
    bar_count=0
    for ax,(task,metric,title) in zip(axes.flat,panels):
        x=np.arange(8)
        for model,offset,color in [('rf',-.19,'#f4f4f4'),('tabpfn',.19,'#377fa6')]:
            data=metrics[(metrics.task==task)&(metrics.metric==metric)&(metrics.model==model)].set_index('element').loc[ELEMENTS]
            bars=ax.bar(x+offset,data['mean'],width=.35,yerr=data.seed_sd,color=color,edgecolor='#424d55',linewidth=.8,
                hatch='//' if model=='rf' else None,error_kw={'elinewidth':.8,'capsize':2,'capthick':.8})
            np.testing.assert_allclose([b.get_height() for b in bars],data['mean']); bar_count+=len(bars)
        if metric=='accuracy':
            ax.scatter(x,data.modal_class_fraction,marker='D',s=20,facecolors='white',edgecolors='#30363c',zorder=4)
        ax.set_title(title,fontsize=13,pad=9)
        ax.set_xticks(x,[e+'†' if task=='coord' and e=='V' else e for e in ELEMENTS],fontsize=12)
        ax.set_axisbelow(True); ax.grid(axis='y',alpha=.18)
        ax.set_ylim(min(0,float(metrics.loc[metrics.metric.isin([p[1] for p in panels]),'mean'].min())-.02),1.02)
        ax.tick_params(axis='y',labelsize=11); ax.spines[['top','right']].set_visible(False)
    for ax in axes[:,0]: ax.set_ylabel('Test score',fontsize=13)
    fig.suptitle('Pointwise XANES test performance: RF and standard TabPFN 3.5',fontsize=17,y=.985)
    fig.legend(handles=[Patch(facecolor='#f4f4f4',edgecolor='#424d55',hatch='//',label='RF'),
        Patch(facecolor='#377fa6',edgecolor='#424d55',label='TabPFN 3.5'),
        Line2D([],[],marker='D',markersize=5,markerfacecolor='white',color='#30363c',ls='',label='Test modal-class fraction (accuracy panel)')],
        loc='upper center',bbox_to_anchor=(.5,.944),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.027,'Bars: matched-seed means ± SD, not sampling confidence intervals. † V coordination: seeds 42 and 44; all others: 42–44.',fontsize=10)
    fig.subplots_adjust(left=.065,right=.99,bottom=.11,top=.84,hspace=.36,wspace=.16)
    assert bar_count==96
    save(fig,FILES[2]); return bar_count


def parity_figure(points,task,number):
    part=points[points.task==task]
    label,unit=('Mean nearest-neighbor distance','Å') if task=='md' else ('Bader charge','e')
    v=part[['reference','prediction']].to_numpy(); lo=float(v.min());hi=float(v.max());pad=.05*(hi-lo)
    limits=(lo-pad,hi+pad)
    fig,axes=plt.subplots(1,2,figsize=(11.6,7.0),sharex=True,sharey=True)
    counts=[]; table=[]
    for ax,model in zip(axes,NAMES):
        maes=[]
        for element in ELEMENTS:
            data=part[(part.model==model)&(part.element==element)]
            plotted=ax.scatter(data.reference,data.prediction,s=18,marker=MARKERS[element],color=COLORS[element],
                alpha=.34,edgecolors=COLORS[element],linewidths=.35,rasterized=False,gid='parity_points')
            np.testing.assert_array_equal(plotted.get_offsets(),data[['reference','prediction']].to_numpy())
            maes.append(float(np.mean(np.abs(data.prediction-data.reference))))
            counts.append({'element':element,'task':task,'model':model,'seed':42,'n':len(data),'mae':maes[-1],
                'r2':float(1-np.sum((data.prediction-data.reference)**2)/np.sum((data.reference-data.reference.mean())**2))})
        table.append([NAMES[model],*[f'{value:.4f}' for value in maes]])
        ax.plot(limits,limits,color='#20252a',ls='--',lw=1.4)
        ax.set(xlim=limits,ylim=limits,aspect='equal')
        ax.set_title(NAMES[model],fontsize=15,pad=9)
        short='distance' if task=='md' else 'charge'
        ax.set_xlabel(f'Reference {short} ({unit})',fontsize=13)
        ax.tick_params(labelsize=11);ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.12)
    axes[0].set_ylabel(f'Predicted {short} ({unit})',fontsize=13)
    fig.suptitle(label+' regression: paired test parity',fontsize=17,y=.99)
    fig.legend(handles=[Line2D([],[],color=COLORS[e],marker=MARKERS[e],ls='',label=e,markersize=7) for e in ELEMENTS],
        loc='upper center',bbox_to_anchor=(.5,.95),ncol=8,frameon=False,fontsize=12)
    tableax=fig.add_axes([.08,.045,.84,.13]);tableax.axis('off')
    t=tableax.table(cellText=table,colLabels=[f'MAE ({unit})',*ELEMENTS],cellLoc='center',loc='center',colWidths=[.19]+[.10125]*8)
    t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,1.4)
    for (row,col),cell in t.get_celld().items():
        cell.set_edgecolor('#dddddd');cell.set_linewidth(.5)
        if row==0: cell.set_facecolor('#f1f2f3');cell.set_text_props(weight='bold')
    fig.text(.08,.202,'All archived test spectra · model seed 42 · identical axes and rows · dashed line: prediction = reference',fontsize=10)
    fig.subplots_adjust(left=.08,right=.98,bottom=.32,top=.83,wspace=.16)
    save(fig,FILES[number])
    return {'task':task,'seed':42,'axis_limits':limits,'counts_and_metrics':counts,'points':len(part)}

def plot_coordination_overview(maps, test_scores, name='figure_01_coordination_comparison'):
    """Fig. 5-style 2×4 overview for the notebook and working paper.

    Contract: eight elements × two models × ten contiguous regions (160 bars),
    using paired validation spectra and identical donor orders within each case.
    Show absorption-maximum distributions above matched permutation effects.
    Shared signed y limits permit magnitude comparison; all bars start at zero.
    Orange/blue encode positive/negative effects, with labels and the zero line
    also conveying sign. Retain the existing green dotted absorption reference
    for consistency with the detailed figures. Export PNG, PDF, SVG and the
    exact plotted numbers. Separate panel headers report held-out test means;
    reduced-seed headers are marked and excluded from primary three-seed claims.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    elements = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
    names = ['Titanium', 'Vanadium', 'Chromium', 'Manganese',
             'Iron', 'Cobalt', 'Nickel', 'Copper']
    panels, bounds = [], []
    for element, full_name in zip(elements, names):
        pair = {m: maps[f'{m}/{element}/coord'] for m in ['rf', 'tabpfn']}
        for field in ['y', 'energy_ev', 'windows', 'donor_orders', 'white_line_indices']:
            np.testing.assert_array_equal(pair['rf'][field], pair['tabpfn'][field])
        d = pair['rf']; energy = d['energy_ev']; edges = energy_edges(energy, d['windows'])
        repeats = pair['tabpfn']['permutation_drops'].shape[1]
        assert repeats in [1, 3, 5], 'Only verified one-, three- or five-shuffle profiles are supported'
        assert d['windows'].shape == (10, 2)
        peak = np.bincount(d['white_line_indices'], minlength=100)/len(d['y'])
        panel = dict(element=element, full_name=full_name, n_validation=len(d['y']),
                     shuffles=repeats, energy_ev=energy.tolist(), region_edges_ev=edges.tolist(),
                     absorption_maximum_fraction=peak.tolist(),
                     modal_maximum_ev=float(energy[peak.argmax()]), models={},
                     test_performance=test_scores[element])
        for model, artifact in pair.items():
            effects = artifact['permutation_drops']
            assert effects.shape == (10, repeats)
            mean, sd = effects.mean(1), effects.std(1, ddof=1 if repeats == 5 else 0)
            bounds.extend([float((mean-sd).min()), float((mean+sd).max())])
            panel['models'][model] = dict(importance=mean.tolist(), shuffle_sd=sd.tolist())
        panels.append(panel)

    ylim = (min(-.025, math.floor(min(bounds)/.025)*.025),
            math.ceil(max(bounds)*1.08/.05)*.05)
    peak_limit = max(max(p['absorption_maximum_fraction']) for p in panels)
    positive, negative, reference = '#C87533', '#357CA5', '#267459'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': name, 'text.color': '#252525',
                         'axes.labelcolor': '#252525'}):
        fig = plt.figure(figsize=(17, 14))
        outer = fig.add_gridspec(2, 4, left=.075, right=.985, bottom=.16,
                                top=.865, wspace=.38, hspace=.23)
        fig.suptitle('Coordination prediction and spectral importance', fontsize=21, y=.978)
        fig.text(.5, .948, 'Panel headers: held-out test means · importance: validation spectra, model seed 42',
                 ha='center', fontsize=12)
        fig.text(.5, .923, 'Importance = original macro-F1 − macro-F1 after shuffling a 10-channel region',
                 ha='center', fontsize=12)
        fig.legend(handles=[Patch(facecolor=positive, edgecolor='#555', label='Positive: prediction worsens'),
                            Patch(facecolor=negative, edgecolor='#555', label='Negative: prediction improves'),
                            Line2D([0], [0], color=reference, ls=':', lw=1.7,
                                   label='Most frequent absorption maximum')],
                   loc='upper center', bbox_to_anchor=(.5, .908), ncol=3, frameon=False, fontsize=11)
        for index, panel in enumerate(panels):
            grid = outer[index//4, index%4].subgridspec(4, 1, height_ratios=[1.18, .45, 1, 1], hspace=.18)
            header = fig.add_subplot(grid[0]); header.set_axis_off()
            scores = panel['test_performance']
            marker = '' if scores['complete'] else '†'
            header.text(0, 1, f"{chr(97+index)}  {panel['full_name']} ({panel['element']}){marker}",
                        fontsize=13, fontweight='bold', va='top')
            header.text(0, .74, f"Test means · n = {scores['n_test']} · {len(scores['matched_seeds'])} seeds{marker}",
                        fontsize=10)
            for x, label in [(0, 'Metric'), (.60, 'RF'), (.91, 'TabPFN')]:
                header.text(x, .53, label, ha='left' if x == 0 else 'center', fontsize=11, fontweight='bold')
            for y, metric, label, scale, precision in [(.33, 'accuracy', 'Accuracy (%)', 100, 1),
                                                      (.13, 'macro_f1', 'Macro-F1', 1, 3)]:
                header.text(0, y, label, fontsize=11)
                for x, model in [(.60, 'rf'), (.91, 'tabpfn')]:
                    header.text(x, y, f"{scale*scores[model]['mean'][metric]:.{precision}f}",
                                ha='center', fontsize=11)
            axes = [fig.add_subplot(grid[i]) for i in range(1, 4)]
            edges = np.asarray(panel['region_edges_ev'])
            axes[0].imshow([panel['absorption_maximum_fraction']], aspect='auto', cmap='Greys',
                           extent=[edges[0], edges[-1], 0, 1], vmin=0, vmax=peak_limit,
                           interpolation='nearest')
            axes[0].set_yticks([])
            axes[0].set_ylabel('Absorption\nmaximum', fontsize=10)
            count = panel['shuffles']
            axes[0].set_title(f"Validation n = {panel['n_validation']} · {count} {'shuffle' if count == 1 else 'shuffles'}",
                              fontsize=9.5, pad=4)
            for ax, (model, label) in zip(axes[1:], [('rf', 'RF'), ('tabpfn', 'TabPFN 3.5')]):
                values = np.asarray(panel['models'][model]['importance'])
                sd = np.asarray(panel['models'][model]['shuffle_sd'])
                bars = ax.bar((edges[:-1]+edges[1:])/2, values, width=np.diff(edges)*.94,
                              color=[positive if v >= 0 else negative for v in values],
                              edgecolor='#555', linewidth=.45,
                              yerr=sd if count > 1 else None,
                              error_kw={'elinewidth': .85, 'capsize': 2, 'ecolor': '#333'})
                np.testing.assert_allclose([b.get_height() for b in bars], values)
                ax.set_ylim(*ylim); ax.set_yticks([0, .1, .2, .3])
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.set_axisbelow(True); ax.grid(axis='y', color='#e8e8e8', lw=.55)
                ax.axhline(0, color='#555', lw=.7)
                ax.set_ylabel(f'{label}\nImportance', fontsize=10)
            for ax in axes:
                ax.axvline(panel['modal_maximum_ev'], color=reference, ls=':', lw=1.5)
                ax.set_xlim(edges[0], edges[-1])
                ax.tick_params(labelsize=10)
                ax.xaxis.set_major_locator(MultipleLocator(10))
                ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            for ax in axes[:-1]: ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel('Energy (eV)', fontsize=11)
        reduced = [p for p in panels if not p['test_performance']['complete']]
        note = 'Test means use seeds 42–44; seed SDs are reported in the supplementary tables.'
        if reduced:
            labels = '; '.join(f"{p['element']}: paired seeds {', '.join(map(str, p['test_performance']['matched_seeds']))}" for p in reduced)
            note += f'\n† {labels} only; descriptive, excluded from the primary three-seed comparison.'
        fig.text(.5, .068, note+'\n'
                 'Gray strips: absorption-maximum distributions. All importance axes share one scale; 0.10 = a 10-percentage-point macro-F1 decrease.\n' +
                 ('All elements: mean ± sample SD of 5 donor shuffles; whiskers measure shuffle variability, not material confidence intervals.'
                  if all(p['shuffles'] == 5 for p in panels) else
                  'Cr: mean ± SD of 3 shuffles (not confidence intervals). Other elements: 1 shuffle; exact rankings remain preliminary.'),
                 ha='center', va='center', fontsize=10.5, linespacing=1.65)
        stem = FIG/name
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, metadata in [('png', {'Software': 'TRIXS importance overview'}),
                                 ('pdf', {'CreationDate': None, 'ModDate': None}),
                                 ('svg', {'Date': None})]:
            fig.savefig(stem.with_suffix('.'+suffix), dpi=200, metadata=metadata)
        plt.close(fig)
    write_json(stem.with_suffix('.json'), dict(
        metric='original macro-F1 minus macro-F1 after regional permutation',
        split='archived validation', model_seed=42, layout='2 rows by 4 columns',
        panel_header_split='archived test',
        panel_header_policy='Means over identical completed seeds for both models; reduced-seed means are descriptive only.',
        importance_ylim=list(ylim), absorption_fraction_vmax=peak_limit,
        colors=dict(positive=positive, negative=negative, modal_maximum=reference), panels=panels))
    return str(stem.with_suffix('.png'))


def collect():
    frame = predictions(); table = metric_table(); summaries=[]; headers={}
    for element in ELEMENTS:
        for task in ['coord','bader','md']:
            part = table[(table.element==element)&(table.task==task)]
            seeds = sorted(set(part[part.model=='rf'].seed)&set(part[part.model=='tabpfn'].seed))
            reference = frame[(frame.element==element)&(frame.task==task)&(frame.model=='rf')&(frame.seed==42)]
            modal = reference.reference.value_counts(normalize=True).max() if task=='coord' else None
            header = dict(n_test=len(reference),matched_seeds=seeds,complete=seeds==[42,43,44])
            for model in NAMES:
                subset = part[(part.model==model)&part.seed.isin(seeds)].sort_values('seed')
                keys = ['accuracy','macro_f1','f1_cn4','f1_cn5','f1_cn6'] if task=='coord' else ['mae','r2']
                header[model] = dict(mean={},sd={})
                for key in keys:
                    values = subset[key].to_numpy()
                    summary = dict(element=element,task=task,model=model,metric=key,
                        mean=float(values.mean()),seed_sd=float(values.std(ddof=0)),
                        seeds=seeds,seed_values=values.tolist(),n_test=len(reference),modal_class_fraction=modal)
                    summaries.append(summary)
                    header[model]['mean'][key]=summary['mean'];header[model]['sd'][key]=summary['seed_sd']
            if task=='coord': headers[element]=header
    maps={}
    for element in ELEMENTS:
        for model in NAMES:
            data,effects = importance(element,model)
            repeats=effects.shape[1]
            data['donor_orders']=data['donor_orders'][:repeats]
            data['permutation_drops']=effects
            maps[f'{model}/{element}/coord']=data
    return dict(metrics=summaries),frame[(frame.seed==42)&(frame.task!='coord')],maps,headers


def build(output_dir=None):
    global FIG
    verify()
    FIG=Path(output_dir) if output_dir is not None else ROOT/'output/figures'
    FIG.mkdir(parents=True,exist_ok=True)
    evidence,points,maps,headers=collect()
    plot_coordination_overview(maps,headers)
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'trixs-prediction-figures'}):
        evidence['performance_bars']=performance_figure(evidence)
        evidence['parity']=[parity_figure(points,'md',3),parity_figure(points,'bader',4)]
    # Numeric reconstruction must agree with the original figure, independently of layout.
    released=json.loads((ROOT/'results/coordination_profiles.json').read_text())
    rebuilt=json.loads((FIG/'figure_01_coordination_comparison.json').read_text())
    for old,new in zip(released['panels'],rebuilt['panels']):
        assert old['element']==new['element']
        for model in NAMES:
            np.testing.assert_allclose(old['models'][model]['importance'],new['models'][model]['importance'],atol=1e-12,rtol=0)
            for metric in ['accuracy','macro_f1']:
                np.testing.assert_allclose(old['test_performance'][model]['mean'][metric],new['test_performance'][model]['mean'][metric],atol=1e-12,rtol=0)
    evidence['sources']={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'results/test_predictions.csv.gz',*sorted((ROOT/'results/importance').glob('*.npz'))]}
    write_json(FIG/'plotted_values.json',evidence)
    return str(FIG)
