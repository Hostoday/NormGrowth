"""Render Figures 4 and 5 from portable saved CSVs in the original paper style.

Run: python -m analysis.plot_core_figures --output-dir build/figures
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
