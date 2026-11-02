"""Scientific lexical tokens with gaps preserved for honest phrase support.

The frozen LDA token lists have already removed stopwords. They cannot prove
that a new bigram was adjacent in the source. This independent representation
starts from immutable normalized Title/Abstract and retains boundary markers.
"""
import re
from pathlib import Path
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .text_processing import normalize_text,formula_spans,TOKEN,lemma
from .batterybert_topic_tools import BOUNDARY

_nltk_data=str(Path(__file__).resolve().parents[1]/'.cache/nltk_data')
if _nltk_data not in nltk.data.path:nltk.data.path.insert(0,_nltk_data)


def lexical_stream(text,phrases=()):
    text=normalize_text(text)
    protected={}
    for i,match in reversed(list(enumerate(formula_spans(text)))):
        marker='chemplaceholder'+chr(97+i//26)+chr(97+i%26)
        protected[marker]=match.group()
        text=text[:match.start()]+' '+marker+' '+text[match.end():]
    text=text.lower();phrase_map={}
    for i,phrase in enumerate(sorted(phrases,key=len,reverse=True)):
        marker='phraseplaceholder'+chr(97+i)
        text=re.sub(r'\b'+r'[\s-]+'.join(map(re.escape,phrase.split()))+r'\b',marker,text)
        phrase_map[marker]=phrase.replace(' ','_')
    tokens=[];previous=0
    def boundary():
        if tokens and tokens[-1]!=BOUNDARY:tokens.append(BOUNDARY)
    for match in TOKEN.finditer(text):
        # Punctuation/slashes and removed words cannot create a new bigram.
        if re.search(r'[^\s]',text[previous:match.start()]):boundary()
        previous=match.end();word=match.group()
        if word in protected:value=protected[word]
        elif word in phrase_map:value=phrase_map[word]
        elif any(marker in word for marker in phrase_map):
            value=re.sub(r'phraseplaceholder[a-z]',lambda m:phrase_map[m.group()],word)
        elif word in ENGLISH_STOP_WORDS or len(word)<=1:
            boundary();continue
        elif word.startswith('chemplaceholder'):raise ValueError('Unresolved chemical placeholder')
        else:value=lemma(word)
        if 'placeholder' in value:raise ValueError('Unresolved scientific placeholder')
        tokens.append(value)
    if tokens and tokens[-1]==BOUNDARY:tokens.pop()
    return tokens


def document_stream(title,abstract,phrases=()):
    first=lexical_stream(title,phrases);second=lexical_stream(abstract,phrases)
    return first+[BOUNDARY]+second if first and second else first or second


def prepare_corpus_stream(out,main,phrases):
    """Cache a versioned lexical representation without rewriting LDA inputs."""
    from pathlib import Path
    import json
    import pandas as pd
    from .final_system_artifacts import ROOT,sha
    from .model_comparison_io import write_json
    folder=Path(out)/'topics';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'scientific_token_stream.parquet';marker=folder/'lexical_stream_identity.json'
    identity=dict(corpus_sha256=sha(ROOT/'data/processed/main_corpus.parquet'),
        source_sha256=sha(Path(__file__)),text_processing_sha256=sha(ROOT/'src/text_processing.py'),
        wordnet_sha256=sha(ROOT/'.cache/nltk_data/corpora/wordnet.zip'),nltk_version=nltk.__version__,phrases=list(phrases))
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf8'))
        if saved['identity']!=identity or sha(path)!=saved['sha256']:raise ValueError('Frozen scientific lexical stream changed')
        frame=pd.read_parquet(path)
        if frame.doc_id.tolist()!=main.doc_id.tolist():raise ValueError('Scientific stream IDs mismatch')
        return frame.tokens.tolist()
    if path.exists():raise ValueError('Scientific stream has no provenance manifest')
    texts=[document_stream(r.title,'' if pd.isna(r.abstract) else r.abstract,phrases) for r in main.itertuples()]
    pd.DataFrame(dict(doc_id=main.doc_id,tokens=texts)).to_parquet(path,index=False)
    write_json(marker,dict(identity=identity,sha256=sha(path),documents=len(main),status='PASS',
        boundary_policy='Preserve gaps from stopwords, punctuation and field changes; native formula/phrase protection'))
    return texts
