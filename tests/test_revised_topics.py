import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from src import revised_topics


def make_catalog(tmp_path,monkeypatch):
    out=tmp_path/'revision';folder=out/'final_topics';folder.mkdir(parents=True);(out/'review').mkdir();(out/'decisions').mkdir()
    (folder/'artifact_manifest.json').write_text('{}')
    pd.DataFrame({'doc_id':['d'+str(i) for i in range(10)],'topic_id':range(10),'source':['Publication']*10,'year':[2020]*10}).to_csv(folder/'topic_assignments.csv',index=False)
    pd.DataFrame({'topic_id':range(10),'topic_label':['name'+str(i) for i in range(10)]}).to_csv(out/'review/topic_labels.csv',index=False)
    (folder/'topic_keywords.csv').write_text('topic_id,rank,token,weight\n')
    (folder/'final_metrics.json').write_text('{}')
    (out/'decisions/geometry.json').write_text('{"accepted":true}')
    monkeypatch.setattr(revised_topics,'OUT',out);monkeypatch.setattr(revised_topics,'CATALOG',folder)
    return out,folder


def test_no_activation_without_valid_gate_and_labels(tmp_path,monkeypatch):
    out,folder=make_catalog(tmp_path,monkeypatch)
    assert revised_topics.available() and not revised_topics.ready()
    marker={'activate_revised_topics':True,'topic_geometry_gate_accepted':True,
       'topic_catalog_manifest_sha256':hashlib.sha256((folder/'artifact_manifest.json').read_bytes()).hexdigest(),
       'topic_label_review_sha256':hashlib.sha256((out/'review/topic_labels.csv').read_bytes()).hexdigest()}
    (out/'deployment.json').write_text(json.dumps(marker));assert revised_topics.ready()
    (out/'review/topic_labels.csv').write_text('tampered')
    assert not revised_topics.ready()


def test_assignment_identity_and_names_are_strict(tmp_path,monkeypatch):
    out,folder=make_catalog(tmp_path,monkeypatch)
    ids=['d'+str(i) for i in range(10)]
    value=revised_topics.assignments(ids)
    assert value.topic_label.tolist()==['name'+str(i) for i in range(10)]
    with pytest.raises(ValueError,match='order mismatch'):revised_topics.assignments(ids[::-1])
    labels=pd.read_csv(out/'review/topic_labels.csv').iloc[:-1];labels.to_csv(out/'review/topic_labels.csv',index=False)
    with pytest.raises(ValueError,match='Missing'):revised_topics.assignments(ids)


def test_real_revised_catalog_has_supported_source_corrected_formula():
    if not revised_topics.available():pytest.skip('Final revision semantic review not yet written')
    frame=revised_topics.assignments();keys=pd.read_csv(revised_topics.CATALOG/'topic_keywords.csv')
    assert len(frame)==13702 and frame.doc_id.is_unique
    assert keys.groupby('topic_id').size().eq(10).all()
    assert 'Na3V2(PO4)3' in set(keys.token)
    assert not keys.token.str.contains('placeholder').any()
    assert (keys.train_document_support>=5).all()
