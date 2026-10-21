"""Stage 2 figures from saved empirical CSV tables only."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS={'Patent':'#2878A5','Publication':'#C76E35'}


def build_figures(tables,figures,dpi=300):
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def read(name): return pd.read_csv(tables/(name+'.csv'),keep_default_na=False)
    def save(name,fig,sources,note=''):
        path=figures/(name+'.png')
        fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        manifest.append(dict(figure=path.relative_to(figures.parents[2]).as_posix(),tables=[(tables/(n+'.csv')).relative_to(figures.parents[2]).as_posix() for n in sources],dpi=dpi,note=note))

    data=read('eda_year_source_counts')
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    for source,group in data.groupby('source'):
        ax.plot(group.year,group.documents,color=COLORS[source],label=source,lw=2)
    ax.set(title='Frozen main corpus: documents by publication year',xlabel='Publication year',ylabel='Documents')
    ax.legend()
    fig.supxlabel('Patent: earliest publication date in PD; Publication: PY. Supplied export ends in 2023; incomplete years and source selection bias apply.',fontsize=8)
    save('eda_01_publication_trends',fig,['eda_year_source_counts'])

    data=read('eda_term_frequencies')
    fig,axes=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    for ax,source in zip(axes,COLORS):
        g=data[data.source.eq(source)].sort_values('document_frequency',ascending=False).head(20).iloc[::-1]
        ax.barh(g.term,g.document_share*100,color=COLORS[source])
        ax.set(title=source,xlabel='Documents containing term (%)')
    fig.suptitle('Top Stage 1 tokens by document frequency (no additional cleaning)')
    save('eda_02_top_terms',fig,['eda_term_frequencies'])

    data=read('eda_ngrams')
    fig,axes=plt.subplots(2,2,figsize=(17,11),layout='constrained')
    for col,source in enumerate(COLORS):
        for row,n in enumerate([2,3]):
            g=data[(data.source==source)&(data.n==n)].head(12).iloc[::-1]
            ax=axes[row,col]
            ax.barh(g.ngram,g.document_share*100,color=COLORS[source])
            ax.tick_params(axis='y',labelsize=8)
            ax.set(title=f'{source}: token {n}-grams',xlabel='Documents containing n-gram (%)')
    fig.suptitle('Most frequent adjacent Stage 1 token sequences')
    fig.supxlabel('Adjacency is measured after Stage 1 preprocessing; sequences may bridge removed stopwords or sentence/title boundaries.',fontsize=8)
    save('eda_03_ngrams',fig,['eda_ngrams'])

    data=read('eda_author_keywords')
    g=data[data.source.eq('Publication')].head(20).iloc[::-1]
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    ax.barh(g.keyword,g.documents,color=COLORS['Publication'])
    ax.set(title='Publication: top author keywords (DE metadata)',xlabel='Documents')
    fig.supxlabel('Patent has no DE field. Keywords are descriptive metadata and are not added to the LDA input.',fontsize=9)
    save('eda_04_author_keywords',fig,['eda_author_keywords'])

    data=read('eda_document_lengths')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for source,g in data.groupby('source'):
        axes[0].hist(g.token_count,bins=np.geomspace(max(1,data.token_count.min()),data.token_count.max()+1,40),density=True,histtype='step',lw=1.8,label=source,color=COLORS[source])
        axes[1].hist(g.abstract_chars[g.abstract_chars>0],bins=np.geomspace(1,data.abstract_chars.max()+1,40),density=True,histtype='step',lw=1.8,label=source,color=COLORS[source])
    axes[0].set(title='LDA token length',xlabel='Tokens (log scale)',ylabel='Density',xscale='log')
    axes[1].set(title='Abstract character length',xlabel='Characters (log scale)',ylabel='Density',xscale='log')
    for ax in axes: ax.legend()
    fig.supxlabel('Densities normalize source size; empty abstracts are retained in the source table but omitted on the log axis.',fontsize=8)
    save('eda_05_length_distributions',fig,['eda_document_lengths'])

    data=read('eda_year_term_trends')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,term in zip(axes.ravel(),['hard_carbon','cathode','anode','electrolyte']):
        for source in COLORS:
            g=data[(data.source==source)&(data.term==term)]
            ax.plot(g.year,g.document_share*100,color=COLORS[source],label=source)
        ax.set(title=term,xlabel='Publication year',ylabel='Documents containing token (%)',ylim=(0,100))
        ax.legend()
    fig.suptitle('Domain term prevalence within each source and year')
    fig.supxlabel('Sparse early-year counts are unstable. Patent year from PD; Publication from PY. No inference about post-export trends.',fontsize=8)
    save('eda_06_domain_term_trends',fig,['eda_year_term_trends'])

    data=pd.read_csv(tables/'lda_comparison.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,(metric,label) in zip(axes.ravel(),[('c_v','C_v coherence (higher is better)'),('u_mass','U_Mass coherence (higher is better)'),('topic_diversity','Top-10 topic diversity'),('stability','Matched top-10 Jaccard stability')]):
        if metric=='stability':
            g=data.dropna(subset=[metric])
            ax.errorbar(g.k,g[metric],yerr=g.stability_std,marker='o',capsize=4,color='#356F56')
            ax.set_ylim(0,1)
        else:
            ax.plot(data.k,data[metric],marker='o',color='#356F56')
        ax.set(title=label,xlabel='Number of topics K',xticks=data.k,xlim=(8,32))
        ax.grid(alpha=.2)
    fig.suptitle('Gensim LDA configuration comparison on the full frozen corpus')
    fig.supxlabel('Base metrics: seed 42. Stability: mean +/- SD of three seed pairs for the two best base C_v configurations only.',fontsize=8)
    save('lda_01_model_comparison',fig,['lda_comparison'])

    data=read('topic_source_distribution')
    counts=data.pivot(index='topic_id',columns='source',values='dominant_documents').fillna(0)
    fig,ax=plt.subplots(figsize=(14,6),layout='constrained')
    bottom=np.zeros(len(counts))
    for source in COLORS:
        ax.bar(counts.index,counts[source],bottom=bottom,color=COLORS[source],label=source)
        bottom+=counts[source].to_numpy()
    ax.set(title='Documents assigned to each dominant LDA topic',xlabel='Topic ID (zero-based)',ylabel='Documents',xticks=counts.index)
    ax.legend()
    save('lda_02_topic_distribution',fig,['topic_source_distribution'])

    share=data.pivot(index='topic_id',columns='source',values='share_of_source_documents').reindex(columns=list(COLORS))
    fig,ax=plt.subplots(figsize=(7,max(6,len(share)*.34)),layout='constrained')
    image=ax.imshow(100*share.values,aspect='auto',cmap='YlGnBu')
    ax.set(xticks=[0,1],xticklabels=share.columns,yticks=range(len(share)),yticklabels=share.index,title='Source-normalized soft topic distribution',ylabel='Topic ID')
    for i in range(len(share)):
        for j in range(2): ax.text(j,i,f'{100*share.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if share.iloc[i,j]>.55*share.values.max() else 'black')
    fig.colorbar(image,ax=ax,label='Expected document share within source (%)')
    save('lda_03_topic_source_heatmap',fig,['topic_source_distribution'],'Each source column sums to 100%; values are soft probability mass, not dominant assignments.')

    trends=read('topic_year_trends')
    years=list(range(int(trends.year.min()),int(trends.year.max())+1))
    k=int(trends.topic_id.max())+1
    fig,axes=plt.subplots(1,2,figsize=(15,max(6,k*.30)),layout='constrained',sharey=True)
    vmax=100*trends.topic_share.max()
    for ax,source in zip(axes,COLORS):
        pivot=trends[trends.source.eq(source)].pivot(index='topic_id',columns='year',values='topic_share').reindex(index=range(k),columns=years)
        image=ax.imshow(np.ma.masked_invalid(pivot.to_numpy()*100),aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmax)
        ticks=[i for i,y in enumerate(years) if y%5==0]
        ax.set(title=source,xlabel='Publication year',xticks=ticks,xticklabels=[years[i] for i in ticks],yticks=range(k),yticklabels=range(k))
        ax.tick_params(axis='x',rotation=45)
    axes[0].set_ylabel('Topic ID')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Within-source/year soft topic share (%)',fraction=.025)
    fig.suptitle('LDA topic trends by source; white columns = no documents')
    fig.supxlabel('One fixed model for all years. Sparse years are descriptive only; Patent PD and Publication PY have different semantics.',fontsize=8)
    save('lda_04_topic_year_heatmap',fig,['topic_year_trends'])

    top=read('topic_summary').sort_values('soft_document_mass',ascending=False).topic_id.head(5).tolist()
    palette=plt.get_cmap('tab10')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for ax,source in zip(axes,COLORS):
        for i,topic in enumerate(top):
            g=trends[(trends.source==source)&(trends.topic_id==topic)]
            ax.plot(g.year,100*g.topic_share,label=f'Topic {topic}',color=palette(i))
        ax.set(title=source,xlabel='Publication year',ylabel='Soft topic share (%)',ylim=(0,100))
        ax.legend(fontsize=8)
    fig.suptitle('Trends of the five largest topics (ranked by total soft document mass)')
    save('lda_05_largest_topic_trends',fig,['topic_summary','topic_year_trends'])

    words=read('topic_keywords')
    chosen=words.groupby('token').weight.sum().sort_values(ascending=False).head(20).index
    pivot=words.pivot(index='topic_id',columns='token',values='weight').reindex(columns=chosen)
    fig,ax=plt.subplots(figsize=(14,max(6,k*.30)),layout='constrained')
    image=ax.imshow(np.ma.masked_invalid(pivot.to_numpy()),aspect='auto',cmap='Blues')
    ax.set(xticks=range(len(chosen)),xticklabels=chosen,yticks=range(k),yticklabels=range(k),ylabel='Topic ID',title='Top-10 keyword weight heatmap: 20 terms with largest summed listed weight')
    ax.tick_params(axis='x',rotation=65,labelsize=8)
    fig.colorbar(image,ax=ax,label='Topic-word probability for listed top-10 terms')
    fig.supxlabel('White cells mean the term is outside that topic\'s top 10; white does not imply a zero model probability.',fontsize=8)
    save('lda_06_keyword_heatmap',fig,['topic_keywords'])
    return manifest
