import pathlib,subprocess,json,os
from source_manifest import manifest,digest
from benchmark_validation import require_ready
root=pathlib.Path(__file__).resolve().parents[4]
out=root/'artifacts/ds-fps/bench/arch-parallel'
built=json.loads((out/'validated-source.json').read_text())
assert built['sourceSha256']==digest(manifest()),'App requires the exact successfully built source.'
require_ready(out)
ordering=json.loads((out/'ordering-summary.json').read_text())
assert ordering['sourceSha256']==built['sourceSha256'] and ordering['rc']==0 and int(ordering['counters']['failed'])==0 and int(ordering['counters']['passed'])>=25 and ordering['requireDevice']=='1','The built source must pass the device ordering tests before packaging.'
revision=(out/'prototype-revision.txt').read_text().strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==revision
assert not subprocess.check_output(['git','status','--porcelain','--','src','tests'],cwd=root,text=True).strip(),'Commit prototype source before packaging.'
env=os.environ.copy();env['SHARPEMU_MAC_GET_TASK_ALLOW']='1'
rc=subprocess.call(['bash','scripts/macos/build-app.sh'],cwd=root,env=env)
if rc==0:
    app=root/'artifacts/macos/SharpEmu.app'
    info=json.loads((app/'Contents/Resources/build-info.json').read_text())
    assert info['source']['revision']==revision and not info['source']['dirtySource']
    subprocess.check_call(['codesign','--verify','--deep','--strict',str(app)])
    (out/'app-ready.json').write_text(json.dumps(dict(sourceSha256=built['sourceSha256'],revision=revision,codeSignatureVerified=True),indent=2)+'\n')
raise SystemExit(rc)
