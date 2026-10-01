"""Render the current manuscript figures from portable saved CSVs.

Run: python -m analysis.plot_core_figures --output-dir build/figures
Manuscript snapshots render full-range Figures 4/5, their restricted sensitivity
views, and the horizontal RQ3 forest plot. Earlier snapshots retain their original
Figure 4/5 selection and plotting behavior.
No fitting, jitter, clipping, or point subsampling is applied.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import _paper_style as style
from . import _paper_style as original
from . import snapshot
from .snapshot import CHECKPOINT, THRESHOLD_COLUMNS, filename, metadata, validate_checkpoints
HERE = Path(__file__).resolve().parents[1] / 'data'
COHORTS = [('Llama','zsRE'),('Llama','CounterFact'),('GPT-2 XL','zsRE'),('GPT-2 XL','CounterFact')]
X_PQ, Y_PQ = (-.60,.30), (.40,2.70)
COLORS={'locality':'#4477AA','rewrite':'#DD8844'}
LABELS={'locality':'Locality prompt','rewrite':'Rewrite prompt'}
METRICS=[('mean_q',r'Mean orthogonal component $\bar q$'),('mean_kappa',r'Mean norm ratio $\bar\kappa_H$'),('mean_abs_norm_deviation',r'Mean absolute norm deviation $D_H$')]
XLIMS={'mean_q':(0.,3.),'mean_kappa':(.8,3.),'mean_abs_norm_deviation':(0.,2.)}
XTICKS={'mean_q':[0,.5,1,1.5,2,2.5,3],'mean_kappa':[1,1.5,2,2.5,3],'mean_abs_norm_deviation':[0,.5,1,1.5,2]}


def visible_limits(default, values):
    """Keep established axes unless new retained observations extend beyond them."""
    padding=.04*(default[1]-default[0])
    minimum,maximum=float(values.min()),float(values.max())
    return (min(default[0],minimum-padding) if minimum<default[0] else default[0],
            max(default[1],maximum+padding) if maximum>default[1] else default[1])


def canvas():
    fig,axes=plt.subplots(2,2,figsize=(9,6.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.983,bottom=.105,top=.825,wspace=.20,hspace=.37)
    for index,(ax,(model,dataset)) in enumerate(zip(axes.flat,COHORTS)):
        ax.set_title(f'({chr(97+index)}) {model} · {dataset}',loc='left',pad=7)
        style.grid(ax)
    style.endpoint_legend(fig)
    return fig,axes


def endpoint_point(ax,x,y,method,editor,size=28):
    color='#bd532e' if method=='Native' else '#565656'
    artist=ax.scatter(x,y,s=size,marker=style.MARKERS[method],
        facecolors=color if editor=='AlphaEdit' else 'white',edgecolors=color,linewidths=.8,zorder=3)
    np.testing.assert_array_equal(artist.get_offsets()[0],[x,y])


def render_endpoint(data):
    fig,axes=canvas(); panels=[]
    x_limits=visible_limits(X_PQ,data.rewrite_mean_p)
    y_limits=visible_limits(Y_PQ,data.rewrite_mean_q)
    for index,(ax,(model,dataset)) in enumerate(zip(axes.flat,COHORTS)):
        part=data[data.model.eq(model)&data.dataset.eq(dataset)]
        assert part.rewrite_mean_p.between(*x_limits).all()
        assert part.rewrite_mean_q.between(*y_limits).all()
        ax.set_xlim(*x_limits);ax.set_ylim(*y_limits)
        ax.set_xticks([-.6,-.4,-.2,0,.2]);ax.set_yticks([.5,1,1.5,2,2.5])
        ax.axvline(0,color='#a6a6a6',lw=.65,ls=(0,(3,3)))
        for row in part.itertuples():
            endpoint_point(ax,row.rewrite_mean_p,row.rewrite_mean_q,row.method,row.editor)
        if index//2==1:ax.set_xlabel(r'Mean parallel component $\bar p_{\mathrm{edit}}$',labelpad=6)
        if index%2==0:ax.set_ylabel(r'Mean orthogonal component $\bar q_{\mathrm{edit}}$',labelpad=6)
        panels.append(dict(model=model,dataset=dataset,visible_points=len(part),x_limits=list(x_limits),y_limits=list(y_limits)))
    stem='fig4_pq_plane'
    style.export(fig,stem,dict(input_points=len(data),visible_points=len(data),off_scale_points=0,
        panels=panels,subplot_layout=[2,2],source=filename('figure4'),
        source_sha256=style.sha(HERE/filename('figure4')),
        geometry_context='rewrite subject_last',geometry_cohort='fixed1000',
        exact_coordinates=True,shared_axes=True,scales='linear',
        removed_endpoint_count=40-len(data)))


def overlay_point(ax,row):
    fill=original.fill_for(row.model,row.dataset,row.order_id)
    size=.82*np.sqrt(15+30*row.edit_count/1000)
    kwargs=original.make_marker(style.MARKERS[row.method],COLORS[row.prompt_context],fill,size)
    kwargs['mew']=.65
    artist,=ax.plot(row.geometry_value,row.locality_percent,**kwargs,
        zorder=4 if fill in ('none','lined') else 3)
    np.testing.assert_array_equal([float(artist.get_xdata()[0]),float(artist.get_ydata()[0])],
                                 [row.geometry_value,row.locality_percent])


def original_bottom_legends(fig, long):
    method=[Line2D([],[],**original.make_marker(style.MARKERS[m],'#626262','full',5.5,m)) for m in style.METHODS]
    fill_specs=[('full','Llama · zsRE'),('right','GPT-2 XL · zsRE'),
                ('none','Llama · CounterFact'),('lined','GPT-2 XL · CounterFact')]
    if long.order_id.ne('canonical').any():
        fill_specs.insert(1, ('left','Llama · zsRE (extra)'))
    cohorts=[Line2D([],[],**original.make_marker('o','#626262',fill,5.5,label)) for fill,label in fill_specs]
    size=[Line2D([],[],**original.make_marker('o','#626262','full',.82*np.sqrt(15+30*n/1000),f'{n:,} edits')) for n in [50,1000]]
    legends=[]
    for handles,y,spacing in [(method,.146,1.9),(cohorts,.093,.95),(size,.038,2.)]:
        legends.append(fig.legend(handles=handles,loc='center',bbox_to_anchor=(.525,y),ncol=len(handles),
            frameon=False,handletextpad=.45,columnspacing=spacing,handlelength=1.,fontsize=9))
    prompt=[Line2D([],[],ls='',marker='o',mfc=COLORS[c],mec=COLORS[c],ms=5,label=LABELS[c]) for c in COLORS]
    fig.legend(handles=prompt,loc='upper center',bbox_to_anchor=(.66,.998),ncol=2,frameon=False,
               handletextpad=.5,columnspacing=1.8,fontsize=9)
    return legends


def main_figure(long, checkpoint_count):
    fig,axes=plt.subplots(2,3,figsize=(9,6.1),sharey=True)
    fig.subplots_adjust(left=.077,right=.986,bottom=.255,top=.91,wspace=.24,hspace=.48)
    panels=[]
    limits={metric:visible_limits(XLIMS[metric],long.loc[long.metric.eq(metric),'geometry_value'])
            for metric,_ in METRICS}
    for ri,dataset in enumerate(['zsRE','CounterFact']):
        for ci,(metric,label) in enumerate(METRICS):
            ax=axes[ri,ci];part=long[long.dataset.eq(dataset)&long.metric.eq(metric)]
            for row in part.sort_values(['edit_count','method','prompt_context'],kind='stable').itertuples():overlay_point(ax,row)
            ax.set_xlim(*limits[metric]);ax.set_xticks(XTICKS[metric]);ax.set_ylim(-1,101);ax.set_yticks([0,20,40,60,80,100])
            ax.set_xlabel(label if ri==1 else '',labelpad=6)
            ax.set_title(f'({chr(97+ri*3+ci)})',loc='left',pad=6,fontsize=10);style.grid(ax)
            assert part.geometry_value.between(*limits[metric]).all()
            assert part.locality_percent.between(0,100).all()
            expected=sorted(part[['geometry_value','locality_percent']].itertuples(index=False,name=None))
            plotted=sorted((float(line.get_xdata()[0]),float(line.get_ydata()[0])) for line in ax.lines)
            assert plotted==expected
            panels.append(dict(dataset=dataset,metric=metric,drawn_points=len(part),
                locality_points=int(part.prompt_context.eq('locality').sum()),rewrite_points=int(part.prompt_context.eq('rewrite').sum()),
                x_limits=list(ax.get_xlim()),y_limits=list(ax.get_ylim()),exact_artist_coordinate_match=True))
        axes[ri,0].set_ylabel('LOC (%)',labelpad=6)
        box=axes[ri,0].get_position();fig.text(box.x0,box.y1+.05,dataset,ha='left',va='bottom',fontsize=10)
    legends=original_bottom_legends(fig, long)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[legend.get_window_extent(renderer) for legend in legends]
    assert all(not boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i))
    assert boxes[0].y1+5<min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[-1])
    unique=long.drop_duplicates('row_id')
    style.export(fig,'fig5_geometry_locality',dict(source=filename('figure5'),
        source_sha256=style.sha(HERE/filename('figure5')),input_checkpoint_count=len(unique),
        dataset_checkpoint_counts=unique.groupby('dataset').size().to_dict(),
        drawn_points=len(long),points_per_metric=len(long)//len(METRICS),
        prompts=['locality','rewrite'],prompt_colors=COLORS,subplot_layout=[2,3],panels=panels,
        geometry_cohort='fixed1000',canonical_orders_only=bool(long.order_id.eq('canonical').all()),
        excluded_checkpoints=checkpoint_count-len(unique),all_retained_points_visible=True,
        same_checkpoint_and_outcome_for_both_prompts=True,method_shapes_preserved=True,
        model_dataset_order_fills_preserved=True,edit_count_marker_size_preserved=True,
        same_metric_axis_limits_across_datasets=True,linear_axes=True))


def manuscript_export(fig, stem, data, audit):
    """Save the plotted coordinates and verify their numerical round trip."""
    style.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = style.OUTPUT_DIR / (stem + '_plotdata.csv')
    data.to_csv(path, index=False)
    reread = pd.read_csv(path, float_precision='round_trip')
    for column in data.select_dtypes(include=np.number).columns:
        np.testing.assert_array_equal(data[column], reread[column])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1
                or box.y1 > fig.bbox.height + 1):
            outside.append(artist.get_text())
    assert not outside, outside
    audit.update(plotdata_file=path.name, plotdata_sha256=style.sha(path),
                 numeric_roundtrip_exact=True, all_text_within_canvas=True,
                 axis_scales=[dict(x=ax.get_xscale(), y=ax.get_yscale(),
                     xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                     xticks=ax.get_xticks().tolist(), yticks=ax.get_yticks().tolist())
                     for ax in fig.axes])
    style.export(fig, stem, audit)


def manuscript_endpoint(data, source, restricted=False):
    """Match the latest full symlog view and its separately named linear view."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    fig.subplots_adjust(left=.105, right=.98, bottom=.115, top=.81,
                        wspace=.3, hspace=.42)
    style.endpoint_legend(fig)
    panels = []
    span = max(float(data.mean_p.max() - data.mean_p.min()), .1)
    for index, (ax, (model, dataset)) in enumerate(zip(axes.flat, COHORTS)):
        part = data[data.model.eq(model) & data.dataset.eq(dataset)]
        expected = (7 if model == 'Llama' else 10) if restricted else 10
        assert len(part) == expected
        ax.set_title(f'({chr(97+index)}) {model} · {dataset} (n = {len(part)})',
                     loc='left', pad=7)
        if restricted:
            ax.set_xlim(data.mean_p.min() - .08*span, data.mean_p.max() + .08*span)
            ax.set_ylim(0, 3)
            ax.set_xticks([-.4, -.2, 0, .2]); ax.set_yticks([0, 1, 2, 3])
        else:
            ax.set_xscale('symlog', linthresh=.5, linscale=3)
            ax.set_yscale('symlog', linthresh=3, linscale=3)
            ax.set_xlim(-.6, data.mean_p.max()*1.5)
            ax.set_ylim(0, data.mean_q.max()*1.5)
            ax.set_xticks([-.5, 0, .5, 1e2, 1e4])
            ax.set_xticklabels(['−0.5', '0', '0.5', r'$10^2$', r'$10^4$'])
            ax.set_yticks([0, 1, 2, 3, 1e2, 1e4, 1e6])
            ax.set_yticklabels(['0', '1', '2', '3', r'$10^2$', r'$10^4$', r'$10^6$'])
            ax.minorticks_off()
        ax.axvline(0, color='#aaa', lw=.6, ls=(0, (3, 3)))
        style.grid(ax)
        for row in part.itertuples():
            color = '#bd532e' if row.method == 'Native' else '#565656'
            artist = ax.scatter(row.mean_p, row.mean_q, s=31,
                marker=style.MARKERS[row.method],
                facecolors=color if row.editor == 'AlphaEdit' else 'white',
                edgecolors=color, lw=.85, zorder=3)
            np.testing.assert_array_equal(artist.get_offsets()[0], [row.mean_p, row.mean_q])
        assert part.mean_p.between(*ax.get_xlim()).all()
        assert part.mean_q.between(*ax.get_ylim()).all()
        if index//2:
            ax.set_xlabel(r'Mean parallel component $\bar p$')
        if index % 2 == 0:
            ax.set_ylabel(r'Mean orthogonal component $\bar q$')
        panels.append(dict(model=model, dataset=dataset, visible_points=len(part)))
    stem = 'fig4_pq_plane' + ('_restricted' if restricted else '')
    manuscript_export(fig, stem, data, dict(source=source,
        source_sha256=style.sha(HERE/source), input_points=len(data),
        canonical_endpoint_count=40, visible_points=len(data), off_scale_points=0,
        removed_endpoint_count=40-len(data), geometry_context='rewrite subject_last',
        geometry_cohort='fixed1000', exact_coordinates=True, shared_axes=True,
        scales='linear' if restricted else 'symlog',
        linthresh_x=None if restricted else .5,
        linthresh_y=None if restricted else 3, linscale=None if restricted else 3,
        subplot_layout=[2, 2], panels=panels))


def manuscript_overlay_legends(fig):
    methods = [Line2D([], [], **style.make_marker(style.MARKERS[method],
        '#626262', 'full', 5.5, method)) for method in style.METHODS]
    cohorts = [Line2D([], [], **style.make_marker('o', '#626262', fill, 5.5, label))
        for fill, label in [('full', 'Llama · zsRE'), ('right', 'GPT-2 XL · zsRE'),
            ('none', 'Llama · CounterFact'), ('lined', 'GPT-2 XL · CounterFact')]]
    sizes = [Line2D([], [], **style.make_marker('o', '#626262', 'full',
        .82*np.sqrt(15+30*n/1000), f'{n:,} edits')) for n in [50, 1000]]
    for handles, y in [(methods, .146), (cohorts, .093), (sizes, .038)]:
        fig.legend(handles=handles, loc='center', bbox_to_anchor=(.525, y),
            ncol=len(handles), frameon=False, columnspacing=1.5,
            handletextpad=.45, handlelength=1, fontsize=9)
    prompts = [Line2D([], [], ls='', marker='o', mfc=COLORS[context],
        mec=COLORS[context], ms=5, label=LABELS[context]) for context in COLORS]
    fig.legend(handles=prompts, loc='upper center', bbox_to_anchor=(.64, .998),
               ncol=2, frameon=False)


def manuscript_overlay(long, source, restricted=False):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.26, top=.9,
                        wspace=.26, hspace=.62)
    panels = []
    for ri, dataset in enumerate(['zsRE', 'CounterFact']):
        n_checkpoints = long.loc[long.dataset.eq(dataset), 'row_id'].nunique()
        for ci, (metric, label) in enumerate(METRICS):
            ax = axes[ri, ci]
            part = long[long.dataset.eq(dataset) & long.metric.eq(metric)]
            assert len(part) == 2*n_checkpoints
            for row in part.sort_values('edit_count').itertuples():
                kwargs = style.make_marker(style.MARKERS[row.method], COLORS[row.prompt_context],
                    style.fill_for(row.model, row.dataset, row.order_id),
                    .82*np.sqrt(15+30*row.edit_count/1000))
                artist, = ax.plot(row.geometry_value, row.locality_percent, **kwargs)
                assert float(artist.get_xdata()[0]) == row.geometry_value
                assert float(artist.get_ydata()[0]) == row.locality_percent
            if restricted:
                ax.set_xlim(0, 3); ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(0, 100)
            else:
                ax.set_xscale('symlog', linthresh=3, linscale=3)
                maximum = long.loc[long.metric.eq(metric), 'geometry_value'].max()
                ax.set_xlim(0, maximum*1.4)
                ax.set_xticks([0, 1, 2, 3, 1e2, 1e4, 1e6])
                ax.set_xticklabels(['0', '1', '2', '3', r'$10^2$', r'$10^4$', r'$10^6$'], fontsize=8)
                ax.minorticks_off(); ax.set_ylim(-1, 101)
            ax.set_yticks([0, 20, 40, 60, 80, 100]); style.grid(ax)
            ax.set_title(f'({chr(97+ri*3+ci)})', loc='left', pad=7)
            if ri == 1:
                ax.set_xlabel(label, labelpad=7)
            if ci == 0:
                ax.set_ylabel('LOC (%)')
            assert part.geometry_value.between(*ax.get_xlim()).all()
            assert part.locality_percent.between(*ax.get_ylim()).all()
            expected = sorted(part[['geometry_value', 'locality_percent']].itertuples(index=False, name=None))
            plotted = sorted((float(line.get_xdata()[0]), float(line.get_ydata()[0])) for line in ax.lines)
            assert plotted == expected
            panels.append(dict(dataset=dataset, metric=metric, checkpoints=n_checkpoints,
                drawn_points=len(part), locality_points=int(part.prompt_context.eq('locality').sum()),
                rewrite_points=int(part.prompt_context.eq('rewrite').sum()), exact_artist_coordinate_match=True))
        box = axes[ri, 0].get_position()
        fig.text(box.x0, box.y1+.065, f'{dataset} ({n_checkpoints} checkpoints)', ha='left', fontsize=10)
    manuscript_overlay_legends(fig)
    unique = long.drop_duplicates('row_id')
    stem = 'fig5_geometry_locality' + ('_restricted' if restricted else '')
    manuscript_export(fig, stem, long, dict(source=source,
        source_sha256=style.sha(HERE/source), input_checkpoint_count=len(unique),
        dataset_checkpoint_counts=unique.groupby('dataset').size().to_dict(),
        drawn_points=len(long), points_per_metric=len(long)//len(METRICS),
        canonical_checkpoint_count=360, excluded_checkpoints=360-len(unique),
        n_trajectories=int(unique.trajectory_id.nunique()), geometry_cohort='fixed1000',
        prompts=['locality', 'rewrite'], prompt_colors=COLORS, subplot_layout=[2, 3], panels=panels,
        canonical_orders_only=True, same_checkpoint_and_outcome_for_both_prompts=True,
        method_shapes_preserved=True, model_dataset_order_fills_preserved=True,
        edit_count_marker_size_preserved=True, exact_coordinates=True,
        same_metric_axis_limits_across_datasets=True, all_retained_points_visible=True,
        scales='linear' if restricted else 'symlog', linear_axes=restricted,
        linthresh=None if restricted else 3, linscale=None if restricted else 3))


def render_manuscript_snapshot():
    """Use the manuscript's full panel; select restricted figures independently."""
    base = HERE / 'manuscript'
    checkpoint_source = str((base/'checkpoints.csv').relative_to(HERE))
    endpoint_source = str((base/'rewrite_endpoints.csv').relative_to(HERE))
    checkpoints = pd.read_csv(HERE/checkpoint_source, float_precision='round_trip')
    endpoint = pd.read_csv(HERE/endpoint_source, float_precision='round_trip')
    validate_checkpoints(checkpoints)
    assert len(checkpoints) == checkpoints.row_id.nunique() == 360
    assert len(endpoint) == endpoint.row_id.nunique() == 40
    assert not endpoint.duplicated(CHECKPOINT).any()
    assert checkpoints.order_id.eq('canonical').all() and endpoint.order_id.eq('canonical').all()
    assert endpoint.edit_count.eq(1000).all() and endpoint.context.eq('rewrite').all()
    assert np.isfinite(checkpoints[THRESHOLD_COLUMNS + ['rewrite_mean_p', 'locality_percent']]).all().all()
    indexed = checkpoints.set_index('row_id')
    assert set(endpoint.row_id) == set(checkpoints.loc[checkpoints.edit_count.eq(1000), 'row_id'])
    for metric in ['mean_p', 'mean_q', 'mean_kappa', 'mean_abs_norm_deviation']:
        np.testing.assert_array_equal(endpoint[metric], indexed.loc[endpoint.row_id, 'rewrite_'+metric])
    keep = checkpoints[THRESHOLD_COLUMNS].le(3).all(axis=1)
    assert int(keep.sum()) == 337
    if 'limit3_sensitivity_retained' in checkpoints:
        saved_keep = checkpoints.limit3_sensitivity_retained.astype(str).str.lower().isin(['true', '1'])
        np.testing.assert_array_equal(keep, saved_keep)
    selected_ids = set(checkpoints.loc[keep, 'row_id'])
    restricted_endpoint = endpoint[endpoint.row_id.isin(selected_ids)].copy()
    assert len(restricted_endpoint) == 34
    identity = CHECKPOINT + ['row_id', 'trajectory_id', 'locality_percent']
    frames = []
    for context in ['locality', 'rewrite']:
        for metric, _ in METRICS:
            column = context+'_'+metric
            part = checkpoints[identity].copy()
            part['prompt_context'] = context
            part['metric'] = metric
            part['source_column'] = column
            part['geometry_value'] = checkpoints[column]
            frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    assert len(long) == 2160 and long.groupby(['row_id', 'metric']).size().eq(2).all()
    restricted_long = long[long.row_id.isin(selected_ids)].copy()
    assert len(restricted_long) == 2022
    manuscript_endpoint(endpoint, endpoint_source)
    manuscript_overlay(long, checkpoint_source)
    manuscript_endpoint(restricted_endpoint, endpoint_source, restricted=True)
    manuscript_overlay(restricted_long, checkpoint_source, restricted=True)
    from .plot_rq3_forest import render as render_forest
    style.AUDITS['rq3_paired_forest_combined'] = render_forest(HERE, style.OUTPUT_DIR)
    return dict(main_endpoints=len(endpoint), main_checkpoints=len(checkpoints),
                main_overlay_points=len(long), restricted_endpoints=len(restricted_endpoint),
                restricted_checkpoints=int(keep.sum()), restricted_overlay_points=len(restricted_long),
                rq3_saved_contrasts=style.AUDITS['rq3_paired_forest_combined']['n_plotted_contrasts'])


def main():
    global HERE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('build/figures'))
    parser.add_argument('--data-dir',type=Path,default=HERE)
    args=parser.parse_args()
    HERE=args.data_dir.resolve()
    snapshot.DATA=HERE
    style.OUTPUT_DIR=args.output_dir.resolve()
    if HERE==style.OUTPUT_DIR or HERE in style.OUTPUT_DIR.parents:
        parser.error('Output must not overwrite the source data directory')
    style.configure()
    style.AUDITS.clear()
    info = metadata()
    if info.get('manuscript_snapshot') or int(info.get('schema_version', 1)) >= 2:
        counts = render_manuscript_snapshot()
        (style.OUTPUT_DIR/'figure_validation.json').write_text(json.dumps(dict(
            passed=True, manuscript_snapshot=True, counts=counts, figures=style.AUDITS), indent=2)+'\n')
        print(json.dumps(counts, indent=2))
        return
    endpoint=pd.read_csv(HERE/filename('figure4'),float_precision='round_trip')
    # Preserve the source grouping and stable within-group draw order.
    endpoint=endpoint.sort_values(['source_group','model','dataset','editor','method'],
        ascending=[False,True,True,True,True],kind='stable')
    assert not endpoint.duplicated(CHECKPOINT).any()
    if metadata().get('canonical_orders_only'):
        assert endpoint.order_id.eq('canonical').all()
    for metric in ['mean_p','mean_q']:
        endpoint['rewrite_'+metric]=endpoint[metric]
    render_endpoint(endpoint)
    long=pd.read_csv(HERE/filename('figure5'),float_precision='round_trip')
    assert len(long)==len(METRICS)*2*long.row_id.nunique()
    assert long.groupby(['row_id','metric']).size().eq(2).all()
    checkpoints=pd.read_csv(HERE/filename('rq2'),float_precision='round_trip')
    validate_checkpoints(checkpoints)
    columns=THRESHOLD_COLUMNS
    retained=checkpoints[checkpoints[columns].le(3).all(axis=1)].set_index('row_id')
    assert set(long.row_id)==set(retained.index)
    for row in long.itertuples():
        assert row.geometry_value == retained.loc[row.row_id,row.source_column]
        assert row.locality_percent == retained.loc[row.row_id,'locality_percent']
    main_figure(long, len(checkpoints))
    (style.OUTPUT_DIR/'figure_validation.json').write_text(json.dumps(dict(passed=True,figures=style.AUDITS),indent=2)+'\n')
    print(f'Rendered Figures 4 and 5: {len(endpoint)} endpoints and {len(long):,} exact-coordinate prompt-overlay points.')

if __name__=='__main__':
    main()
