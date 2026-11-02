from src.full_document_inference import query_windows


class TokenizerFixture:
    cls_token_id=101
    sep_token_id=102
    def __call__(self,text,**kwargs):
        assert kwargs['truncation'] is False
        return dict(input_ids=list(range(1000,1000+len(text.split()))))
    def num_special_tokens_to_add(self,pair=False):return 2


def test_long_query_all_tokens_and_tail_are_kept():
    tokenizer=TokenizerFixture()
    windows,count=query_windows(tokenizer,' '.join(['token']*1803))
    assert count==1803 and len(windows)>1
    assert sum(r['unique_tokens'] for r in windows)==1803
    assert windows[-1]['token_end']==1803
    assert 2802 in windows[-1]['input_ids']
    assert all(len(r['input_ids'])<=256 for r in windows)


def test_short_query_matches_native_template_without_padding_or_retokenization():
    windows,count=query_windows(TokenizerFixture(),'hard carbon anode')
    assert count==3 and len(windows)==1
    assert windows[0]['input_ids']==[101,1000,1001,1002,102]
