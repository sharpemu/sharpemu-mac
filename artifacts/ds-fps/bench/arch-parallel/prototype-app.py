import pathlib,subprocess,json,os
from source_manifest import manifest,digest
root=pathlib.Path(__file__).resolve().parents[4]
out=root/'artifacts/ds-fps/bench/arch-parallel'
built=json.loads((out/'validated-source.json').read_text())
assert built['sourceSha256']==digest(manifest()),'App requires the exact successfully built source.'
ordering=json.loads((out/'ordering-summary.json').read_text())
assert ordering['sourceSha256']==built['sourceSha256'] and ordering['rc']==0 and int(ordering['counters']['failed'])==0 and int(ordering['counters']['passed'])>=25 and ordering['requireDevice']=='1','The built source must pass the device ordering tests before packaging.'
revision=(out/'prototype-revision.txt').read_text().strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==revision
assert not subprocess.check_output(['git','status','--porcelain','--','src','tests'],cwd=root,text=True).strip(),'Commit prototype source before packaging.'
env=os.environ.copy();env['SHARPEMU_MAC_GET_TASK_ALLOW']='1'
raise SystemExit(subprocess.call(['bash','scripts/macos/build-app.sh'],cwd=root,env=env))
