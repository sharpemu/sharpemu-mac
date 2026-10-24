import sys,os,pathlib,subprocess,json,datetime,xml.etree.ElementTree as ET
from source_manifest import manifest,digest
root=pathlib.Path(__file__).resolve().parents[4]
out=root/'artifacts/ds-fps/bench/arch-parallel'
built=json.loads((out/'validated-source.json').read_text())
assert built['files']==manifest(), 'Tests require the exact successfully built prototype source.'
name=sys.argv[1]
projects=dict(metal='SharpEmu.Gpu.Metal.Tests',libs='SharpEmu.Libs.Tests',shader='SharpEmu.ShaderCompiler.Tests',msl='SharpEmu.ShaderCompiler.Metal.Tests',ordering='SharpEmu.Gpu.Metal.Tests')
project=projects[name]
results=out/'tests'/name/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
results.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.pop('SHARPEMU_TEST_REQUIRE_DEVICE',None)
env['SHARPEMU_METAL_V2_PARALLEL_ENCODE']='0'
env['SHARPEMU_METAL_V2_BATCH_CENSUS']='0'
if name in ('metal','msl','ordering'):env['SHARPEMU_TEST_REQUIRE_DEVICE']='1'
cmd=[str(pathlib.Path.home()/'.dotnet-x64/dotnet'),'test',str(root/'tests'/project/(project+'.csproj')),'-c','Release','--no-build','--no-restore','--blame-hang-timeout','120s','--blame-hang-dump-type','mini','--results-directory',str(results),'--logger','trx;LogFileName='+name+'.trx','-p:RestorePackagesPath='+str(pathlib.Path.home()/'.nuget/sharpemu-shared'),'-nodeReuse:false','-p:UseSharedCompilation=false']
if name=='shader':cmd+=['--settings','/Users/xnetcat/Projects/sharpemu/sharpemu-mac/artifacts/ds-fps/shader-serial.runsettings']
if name=='ordering':cmd+=['--filter','FullyQualifiedName~MetalParallelBatchTests|FullyQualifiedName~ParallelEncodeSceneTests']
print(json.dumps(dict(command=cmd,requireDevice=env.get('SHARPEMU_TEST_REQUIRE_DEVICE'),parallelDefault=env['SHARPEMU_METAL_V2_PARALLEL_ENCODE']),indent=2),flush=True)
rc=subprocess.call(cmd,cwd=root,env=env)
trx=results/(name+'.trx')
if trx.exists():
    tree=ET.parse(trx);ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    counters=tree.find('.//t:Counters',ns)
    rows=[dict(test=x.attrib.get('testName'),outcome=x.attrib.get('outcome'),message=x.findtext('t:Output/t:ErrorInfo/t:Message',default='',namespaces=ns)) for x in tree.findall('.//t:UnitTestResult',ns) if x.attrib.get('outcome')!='Passed']
    summary=dict(rc=rc,sourceSha256=built['sourceSha256'],command=cmd,requireDevice=env.get('SHARPEMU_TEST_REQUIRE_DEVICE'),trx=str(trx),counters=counters.attrib if counters is not None else None,nonpassing=rows)
    payload=json.dumps(summary,indent=2)+'\n'
    (out/(name+'-summary.json')).write_text(payload)
    (results/'summary.json').write_text(payload)
raise SystemExit(rc)
