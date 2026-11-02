from src.scientific_keyword_stream import document_stream,lexical_stream
from src.batterybert_optimization import ContiguousScientificVectorizer


def terms(title,abstract='',phrases=()):
    stream=document_stream(title,abstract,phrases)
    return set(ContiguousScientificVectorizer(ngram_range=(1,2),lowercase=False).build_analyzer()(' '.join(stream)))


def test_removed_stopwords_punctuation_and_field_boundaries_do_not_invent_phrases():
    assert 'hard carbon' not in terms('hard and carbon')
    assert 'hard carbon' not in terms('hard. Carbon')
    assert 'hard carbon' not in terms('hard / carbon')
    assert 'hard carbon' not in terms('hard','carbon')
    assert 'hard carbon' in terms('hard carbon')


def test_protected_chemical_formulas_and_true_hyphenated_phrases_survive():
    tokens=lexical_stream('Na3V2(PO4)3 NaFePO4 Na+ all-solid-state hard-carbon anodes',
        ['solid state','hard carbon'])
    assert {'Na3V2(PO4)3','NaFePO4','Na+','all-solid_state','hard_carbon','anode'}<=set(tokens)
    assert not any('placeholder' in t for t in tokens)


def test_lemmatization_cannot_join_words_across_removed_function_words():
    value=terms('Electrodes were storing sodium in porous carbon.')
    assert {'electrode','store','sodium','porous carbon'}<=value
    assert 'electrode store' not in value and 'sodium porous' not in value


def test_lexical_cli_uses_project_wordnet_without_pytest_resource_setup(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code="import sys; sys.path.insert(0,sys.argv[1]); from src.scientific_keyword_stream import lexical_stream; assert 'electrode' in lexical_stream('electrodes')"
    result=subprocess.run([sys.executable,'-c',code,str(root)],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
