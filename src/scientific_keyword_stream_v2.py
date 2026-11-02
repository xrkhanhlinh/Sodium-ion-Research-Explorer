"""Versioned scientific tokens preserving complete nested chemical notation.

This module does not modify the historical text_processing or lexical stream.
Canonicalization is restricted to unspaced numeric parentheses used as subscripts, e.g.
Na(3)V(2)(PO(4))(3) -> Na3V2(PO4)3. Element case, coefficients, group structure
and charge signs are never inferred or chemically repaired. Provenance retains
every original span and applied transformation. Syntax recognition is not a
claim that a composition exists or is valid for sodium-ion batteries.
"""
from dataclasses import dataclass
import re
from pathlib import Path
import nltk

from .text_processing import normalize_text,TOKEN,lemma
from .batterybert_topic_tools import BOUNDARY
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ELEMENTS=frozenset('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())
_CANDIDATE=re.compile(r'(?<![A-Za-z0-9])(\[*[A-Z][A-Za-z0-9δ().\[\]]*)[+-]?')
_NUMERIC_PAREN=re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*\)')
_COUNT=re.compile(r'(?:\d+(?:\.\d+)?|[xyzδ])')

for _parent in Path(__file__).resolve().parents:
    _data=_parent/'.cache/nltk_data'
    if (_data/'corpora/wordnet.zip').is_file():
        if str(_data) not in nltk.data.path:nltk.data.path.insert(0,str(_data))
        break


@dataclass(frozen=True)
class FormulaOccurrence:
    start:int
    end:int
    raw:str
    canonical:str
    actions:tuple
    syntax_recognized:bool


def _valid_formula(text):
    """Recognize a conservative element/group grammar, including nested groups."""
    if not re.search(r'\d|[+\-]|[xyzδ]',text):return False
    body=text[:-1] if text[-1:] in ('+','-') else text
    def count(at):
        found=_COUNT.match(body,at)
        return found.end() if found else at
    def sequence(at,inside=False):
        parts=0
        while at<len(body):
            if body[at]==')':return at,parts>0
            if body[at]=='(':
                end,valid=sequence(at+1,True)
                if not valid or end>=len(body) or body[end]!=')':return at,False
                at=count(end+1);parts+=1;continue
            found=re.match(r'[A-Z][a-z]?',body[at:])
            if not found or found.group() not in ELEMENTS:return at,False
            at=count(at+len(found.group()));parts+=1
        return at,parts>0 and not inside
    end,valid=sequence(0)
    return valid and end==len(body)


def _numeric_subscripts(raw):
    """Convert only a numeric group after an uncounted element/chemical group.

    TiO2(101) is potentially a Miller index and must never become TiO2101.
    Whitespace-separated annotations are not included in a formula span at all.
    This syntax-only rule cannot establish chemical identity or validity.
    """
    converted=[];previous=0
    for match in _NUMERIC_PAREN.finditer(raw):
        prefix=raw[:match.start()]
        allowed=False
        element=re.search(r'[A-Z][a-z]?$',prefix)
        if element and element.group() in ELEMENTS:
            allowed=True
        elif prefix.endswith(')'):
            depth=0;opening=None
            for at in range(len(prefix)-1,-1,-1):
                if prefix[at]==')':depth+=1
                elif prefix[at]=='(':
                    depth-=1
                    if depth==0:opening=at;break
            # A preceding numeric annotation such as Na(3)(2) cannot be joined
            # into the invented coefficient Na32.
            allowed=opening is not None and bool(re.search(r'[A-Z]',prefix[opening+1:-1]))
        converted.extend([raw[previous:match.start()],match.group(1) if allowed else match.group()])
        previous=match.end()
    converted.append(raw[previous:])
    return ''.join(converted)


def formula_occurrences(text):
    """Return complete source spans; unsupported notation is retained, not split."""
    consumed_until=0
    for match in _CANDIDATE.finditer(text):
        if match.start()<consumed_until:continue
        raw=match.group();end=match.end()
        # Never accept a valid-looking prefix of unsupported variable/fractional
        # or superscript notation as if it were a complete composition/charge.
        while end<len(text) and re.match(r'[A-Za-z0-9δ().\[\]+\-/⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]',text[end]):end+=1
        raw=text[match.start():end];consumed_until=end
        # Sentence punctuation and enclosing prose parentheses are not formula.
        while raw.endswith('.'):
            raw=raw[:-1];end-=1
        while raw.endswith(')') and raw.count(')')>raw.count('('):
            raw=raw[:-1];end-=1
        if not re.search(r'\d|[+\-⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]',raw):continue
        canonical=_numeric_subscripts(raw)
        actions=() if canonical==raw else ('numeric_parentheses_to_subscript',)
        valid=_valid_formula(canonical)
        if actions and len(re.findall(r'[A-Z][a-z]?',canonical))<2:
            # Bare Na(1), Na(2), Fe(2) may denote a crystal site or oxidation
            # state. Without a multi-element composition, keep source literal.
            canonical=raw;actions=();valid=False
        if not valid:
            head=re.match(r'\[*([A-Z][a-z]?)',raw)
            if not head or head.group(1) not in ELEMENTS:continue
            remainder=raw[head.end():]
            if not remainder or not re.match(r'[A-Z0-9(\[xyzδ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]',remainder):continue
        yield FormulaOccurrence(match.start(),end,raw,canonical if valid else raw,actions if valid else (),valid)


def lexical_stream_v2(text,phrases=(),return_provenance=False):
    text=normalize_text(text)
    occurrences=list(formula_occurrences(text));protected={}
    for i,item in reversed(list(enumerate(occurrences))):
        marker='chemvtoken'+chr(97+i//26)+chr(97+i%26)
        protected[marker]=item.canonical
        text=text[:item.start]+' '+marker+' '+text[item.end:]
    text=text.lower();phrase_map={}
    for i,phrase in enumerate(sorted(phrases,key=len,reverse=True)):
        marker='phrasevtoken'+chr(97+i//26)+chr(97+i%26)
        text=re.sub(r'\b'+r'[\s-]+'.join(map(re.escape,phrase.split()))+r'\b',marker,text)
        phrase_map[marker]=phrase.replace(' ','_')
    tokens=[];previous=0
    def boundary():
        if tokens and tokens[-1]!=BOUNDARY:tokens.append(BOUNDARY)
    for match in TOKEN.finditer(text):
        if re.search(r'[^\s]',text[previous:match.start()]):boundary()
        previous=match.end();word=match.group()
        if word in protected:value=protected[word]
        elif word in phrase_map:value=phrase_map[word]
        elif any(marker in word for marker in phrase_map):
            value=word
            for marker,phrase in phrase_map.items():value=value.replace(marker,phrase)
        elif word in ENGLISH_STOP_WORDS or len(word)<=1:
            boundary();continue
        else:value=lemma(word)
        if 'chemvtoken' in value or 'phrasevtoken' in value:raise ValueError('Unresolved versioned token')
        tokens.append(value)
    if tokens and tokens[-1]==BOUNDARY:tokens.pop()
    if return_provenance:return tokens,[item.__dict__ for item in occurrences]
    return tokens


def document_stream_v2(title,abstract,phrases=()):
    a=lexical_stream_v2(title,phrases);b=lexical_stream_v2(abstract,phrases)
    return a+[BOUNDARY]+b if a and b else a or b
