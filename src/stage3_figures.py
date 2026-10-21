"""Publication figures from persisted Stage 3 tables, without neural imports."""
import os
from .stage3_common import ROOT,REPORT,TABLE,FIGURE,read_json,write_json,progress
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    manifest=[]
    def save(fig,name,tables,note):
        fig.tight_layout();fig.savefig(FIGURE/name,dpi=300,bbox_inches='tight');plt.close(fig)
        manifest.append(dict(figure=(FIGURE/name).relative_to(ROOT).as_posix(),tables=[(TABLE/t).relative_to(ROOT).as_posix() for t in tables],note=note,dpi=300))
    loss=pd.read_csv(TABLE/'training_loss.csv');val=pd.read_csv(TABLE/'validation_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(loss.step,loss.loss,alpha=.25,color='#1d719a',lw=.6)
    axes[0].plot(loss.step,loss.loss.rolling(25,min_periods=1).mean(),color='#1d719a',label='25-step rolling mean')
    axes[0].set(xlabel='Optimizer step',ylabel='Symmetric contrastive loss',title='Real final-block BGE fine-tuning');axes[0].legend()
    axes[1].plot(val.step,val.group_mrr_at_10,marker='o',color='#d47728')
    best=read_json(ROOT/'models/bert_finetuned/training_success.json')['best_step']
    axes[1].axvline(best,ls='--',color='gray',label=f'Selected step {best}');axes[1].legend()
    axes[1].set(xlabel='Optimizer step',ylabel='Group MRR@10',title='Validation only checkpoint selection',ylim=(0,1))
    save(fig,'bert_01_training.png',['training_loss.csv','validation_metrics.csv'],'All real optimizer-step losses; validation only, not test.')
    metrics=pd.read_csv(TABLE/'bert_before_after.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4));columns=['group_mrr_at_10','group_recall_at_1','group_recall_at_5','group_recall_at_10']
    for ax,split in zip(axes,['validation','test']):
        sub=metrics[metrics.split.eq(split)].set_index('phase');x=np.arange(4)
        for phase,offset,color in [('pretrained',-.18,'#7895b2'),('finetuned',.18,'#d47728')]:
            values=sub.loc[phase,columns].astype(float).to_numpy();ax.bar(x+offset,values,width=.35,label=phase,color=color)
        ax.set(xticks=x,xticklabels=['MRR@10','Success@1','Success@5','Success@10'],ylim=(0,1),title=split.capitalize()+' title-to-abstract proxy',ylabel='Group-aware metric')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    save(fig,'bert_02_before_after.png',['bert_before_after.csv'],'Fixed held-out splits and candidates; not an SE evaluation.')
    topics=pd.read_csv(TABLE/'bertopic_topic_summary.csv');real=topics[~topics.is_outlier].sort_values('documents',ascending=False)
    fig,ax=plt.subplots(figsize=(12,4.8));x=np.arange(len(topics));ordered=topics.sort_values('documents',ascending=False)
    ax.bar(x,ordered.documents,color=['#999999' if t==-1 else '#1d719a' for t in ordered.topic_id]);step=max(1,len(x)//35)
    ax.set(xticks=x[::step],xticklabels=[str(t) for t in ordered.topic_id.iloc[::step]],xlabel='Topic ID (sorted by count; -1 = outliers)',ylabel='Assigned documents',title='BERTopic distribution: data-derived topics and outliers')
    save(fig,'bertopic_01_distribution.png',['bertopic_topic_summary.csv'],'Counts include all outliers; no outlier reassignment.')
    keywords=pd.read_csv(TABLE/'bertopic_keywords.csv',keep_default_na=False);top=real.head(12)
    columns=3;rows=max(1,int(np.ceil(len(top)/columns)));fig,axes=plt.subplots(rows,columns,figsize=(15,3.2*rows),squeeze=False)
    for ax in axes.ravel():ax.set_visible(False)
    for ax,row in zip(axes.ravel(),top.itertuples()):
        ax.set_visible(True);sub=keywords[keywords.topic_id.eq(row.topic_id)].head(10).iloc[::-1]
        ax.barh(sub.token,sub.ctfidf_weight,color='#1d719a');ax.set_title(f'Topic {row.topic_id}: n={row.documents:,}');ax.set_xlabel('c-TF-IDF weight');ax.tick_params(axis='y',labelsize=9)
    save(fig,'bertopic_02_keywords.png',['bertopic_keywords.csv','bertopic_topic_summary.csv'],'Top-10 words of up to 12 largest non-outlier topics; all words in CSV.')
    similarity=pd.read_csv(TABLE/'bertopic_similarity.csv');matrix=similarity.drop(columns='topic_id').to_numpy();ids=similarity.topic_id.tolist()
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(matrix,vmin=float(matrix.min()),vmax=1,cmap='viridis');stride=max(1,int(np.ceil(len(ids)/35)));ticks=np.arange(0,len(ids),stride)
    ax.set(xticks=ticks,yticks=ticks,xticklabels=[ids[i] for i in ticks],yticklabels=[ids[i] for i in ticks],xlabel='Topic ID',ylabel='Topic ID',title='Cosine similarity of fine-tuned embedding centroids')
    fig.colorbar(im,ax=ax,label='Cosine similarity',ticks=np.linspace(float(matrix.min()),1,6),format='%.2f');save(fig,'bertopic_03_similarity_heatmap.png',['bertopic_similarity.csv'],'Non-outlier semantic centroids; not c-TF-IDF similarity; negative values retained.')
    points=pd.read_csv(TABLE/'bertopic_umap_2d.csv');fig,ax=plt.subplots(figsize=(11,7));out=points.topic_id.eq(-1)
    if out.any():ax.scatter(points.loc[out,'x'],points.loc[out,'y'],s=3,color='#b5b5b5',alpha=.35,label='Outliers (-1)',rasterized=True)
    valid=points[~out];n=max(1,len(ids));cmap=plt.get_cmap('turbo',n)
    sc=ax.scatter(valid.x,valid.y,c=valid.topic_id,s=4,cmap=cmap,vmin=-.5,vmax=n-.5,alpha=.7,rasterized=True)
    fig.colorbar(sc,ax=ax,label='Topic ID');ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title='Fine-tuned BGE document embeddings: separate 2-D projection')
    if out.any():ax.legend(loc='upper right')
    save(fig,'bertopic_04_umap_2d.png',['bertopic_umap_2d.csv'],'Visualization UMAP is separate from five-dimensional clustering UMAP; all documents plotted.')
    comparison=pd.read_csv(TABLE/'lda_vs_bertopic.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,label in zip(axes,['c_v','topic_diversity','outlier_ratio'],['C_v (shared token reference)','Top-10 Topic Diversity','Outlier ratio']):
        ax.bar(['LDA','BERTopic'],comparison[metric],color=['#7895b2','#d47728']);ax.set(title=label,ylim=(0,1))
    save(fig,'comparison_01_topics.png',['lda_vs_bertopic.csv'],'Different clustering objectives/counts; no direct stability bar because methods/runs differ.')
    write_json(REPORT/'figure_manifest.json',manifest);progress(f'Saved {len(manifest)} Stage 3 figures at 300 DPI with source tables. Tests and completion report pending.')

if __name__=='__main__':run()
