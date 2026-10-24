import json
from source_manifest import manifest,digest
def require_ready(out):
    built=json.loads((out/'validated-source.json').read_text())
    assert built['sourceSha256']==digest(manifest()),'Game requires the exact built and tested source.'
    known=json.loads((out/'known-base-failures.json').read_text())['failures']
    checked={}
    for suite in ('ordering','metal','libs','shader','msl'):
        result=json.loads((out/(suite+'-summary.json')).read_text())
        assert result['sourceSha256']==built['sourceSha256'],(suite,'stale test result')
        counters=result['counters']
        assert int(counters['executed'])>0 and int(counters.get('aborted',0))==0,(suite,counters)
        failed={item['test'] for item in result['nonpassing'] if item['outcome']=='Failed'}
        unexpected=failed-set(known.get(suite,[]))
        assert not unexpected,(suite,'new failures',sorted(unexpected))
        assert int(counters['failed'])==len(failed),(suite,'failure accounting')
        assert result['rc']==(1 if failed else 0),(suite,'incomplete or unexpected runner result')
        if suite=='ordering':
            assert int(counters['passed'])>=24 and result['requireDevice']=='1',(suite,'device ordering cases missing')
        checked[suite]=dict(counters=counters,knownBaseFailures=sorted(failed),trx=result['trx'])
    (out/'benchmark-validation.json').write_text(json.dumps(dict(sourceSha256=built['sourceSha256'],suites=checked),indent=2)+'\n')
