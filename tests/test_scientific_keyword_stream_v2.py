"""Adversarial chemical notation and phrase evidence contracts for v2 only."""
from src.scientific_keyword_stream_v2 import formula_occurrences,lexical_stream_v2,document_stream_v2
from src.batterybert_topic_tools import BOUNDARY


def normalized(text):return [x.canonical for x in formula_occurrences(text)]


def test_numeric_parentheses_are_unambiguous_source_equivalences():
    variants=['Na3V2(PO4)3','Na3V2(PO4)(3)','Na(3)V(2)(PO(4))(3)']
    for text in variants:
        found=list(formula_occurrences(text))
        assert len(found)==1
        assert found[0].canonical=='Na3V2(PO4)3'
        assert found[0].raw==text
        assert found[0].syntax_recognized


def test_complete_compositions_groups_and_charges_are_preserved():
    values=['Na3V2(PO4)2F3','Fe(CN)(6)','Na+','Fe3+','NaFePO4','CoO2','CO2']
    assert [normalized(text)[0] for text in values]==['Na3V2(PO4)2F3','Fe(CN)6','Na+','Fe3+','NaFePO4','CoO2','CO2']
    assert normalized('CoO2')!=normalized('CO2')


def test_no_truncated_nested_formula_or_unbalanced_enclosing_parentheses():
    assert normalized('(Na(3)V(2)(PO(4))(3)).')==['Na3V2(PO4)3']
    found=list(formula_occurrences('Na3V2(PO4)(oops)'))
    assert len(found)==1 and not found[0].syntax_recognized
    assert found[0].canonical=='Na3V2(PO4)(oops)'
    assert normalized('Na3V2(PO4)(oops)')!=['Na3V2(PO4)']


def test_no_invented_phrase_across_gaps_fields_or_stopwords():
    assert lexical_stream_v2('hard and carbon')==['hard',BOUNDARY,'carbon']
    assert lexical_stream_v2('hard. Carbon')==['hard',BOUNDARY,'carbon']
    assert document_stream_v2('hard','carbon')==['hard',BOUNDARY,'carbon']
    assert 'hard_carbon' in lexical_stream_v2('hard-carbon anodes',['hard carbon'])


def test_provenance_retains_original_stoichiometry_without_chemical_invention():
    tokens,evidence=lexical_stream_v2('Na3V2(PO4)(3) and Na3V2(PO4)2F3',return_provenance=True)
    assert 'Na3V2(PO4)3' in tokens and 'Na3V2(PO4)2F3' in tokens
    assert evidence[0]['raw']=='Na3V2(PO4)(3)'
    assert evidence[0]['actions']==('numeric_parentheses_to_subscript',)
    assert evidence[1]['actions']==()


def test_unsupported_variable_fractional_or_superscript_notation_is_never_truncated():
    for raw in ['Na1-xCoO2','Na(2/3)Ni(1/3)Mn(2/3)O2','Fe³⁺']:
        found=list(formula_occurrences(raw))
        assert len(found)==1 and found[0].raw==raw
        assert found[0].canonical==raw
        assert not found[0].syntax_recognized
        assert raw in lexical_stream_v2(raw)


def test_numbered_ordinary_words_are_not_protected_as_formula_tokens():
    assert list(formula_occurrences('Table (3) Figure2 Carbon (4) Room (3)'))==[]


def test_miller_indices_whitespace_annotations_and_phase_names_are_never_compositions():
    assert normalized('TiO2 (101) facets')==['TiO2']
    assert normalized('Na3V2(PO4) (3)')==['Na3V2(PO4)']
    for raw in ['TiO2(101)','P2(1)ma','P6(3)/mmc','Na(3)(2)']:
        found=list(formula_occurrences(raw))
        assert len(found)==1
        assert found[0].raw==raw and found[0].canonical==raw
        assert not found[0].syntax_recognized


def test_bare_element_site_labels_are_not_invented_stoichiometric_formulas():
    for raw in ['Na(1)','Na(2)','Fe(2)']:
        found=list(formula_occurrences(raw))
        assert len(found)==1 and found[0].canonical==raw
        assert not found[0].syntax_recognized and found[0].actions==()


def test_square_bracket_complexes_are_retained_whole_without_publishing_interior_fragments():
    for raw in ['Fe[Fe(CN)(6)]','Na4[Fe(CN)6]','[Fe(CN)6]3-']:
        found=list(formula_occurrences(raw))
        assert len(found)==1 and found[0].raw==raw
        assert found[0].canonical==raw and not found[0].syntax_recognized
        assert raw in lexical_stream_v2(raw)
