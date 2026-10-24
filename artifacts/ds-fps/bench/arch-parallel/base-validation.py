import sys,pathlib,subprocess,os,json,hashlib,xml.etree.ElementTree as ET
root=pathlib.Path(__file__).resolve().parents[4]
out=root/'artifacts/ds-fps/bench/arch-parallel'
base=root/'artifacts/arch-parallel-base-validation'
revision='bdc7ea28c8e7df0b8753966d8ba1881f60e5e764'
project=base/'tests/SharpEmu.Gpu.Metal.Tests/SharpEmu.Gpu.Metal.Tests.csproj'
mode=sys.argv[1]
env=os.environ.copy();env.update(MSBUILDDISABLENODEREUSE='1',SHARPEMU_TEST_REQUIRE_DEVICE='1',SHARPEMU_METAL_V2_PARALLEL_ENCODE='0',SHARPEMU_METAL_V2_BATCH_CENSUS='0')
cmd=[str(pathlib.Path.home()/'.dotnet-x64/dotnet'),mode,str(project),'-c','Release','-p:RestorePackagesPath='+str(pathlib.Path.home()/'.nuget/sharpemu-shared'),'-nodeReuse:false','-p:UseSharedCompilation=false']
results=out/'tests/base-guest-views'
if mode=='test':
    results.mkdir(parents=True,exist_ok=True)
    cmd+=['--no-build','--no-restore','--blame-hang-timeout','120s','--blame-hang-dump-type','mini','--filter','FullyQualifiedName~MetalGuestViewTests|FullyQualifiedName~MetalLossyTextureTests','--results-directory',str(results),'--logger','trx;LogFileName=base.trx']
# Assert every archived file is exactly the selected base blob before compilation/execution.
for row in subprocess.check_output(['git','ls-tree','-r',revision],cwd=root,text=True).splitlines():
    info,name=row.split('\t',1);modebits,kind,sha=info.split()
    if kind!='blob':continue
    data=(base/name).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert actual==sha,name
print(json.dumps(dict(base=revision,command=cmd,requireDevice=env['SHARPEMU_TEST_REQUIRE_DEVICE']),indent=2),flush=True)
rc=subprocess.call(cmd,cwd=base,env=env)
record=dict(base=revision,command=cmd,rc=rc,archiveFilesMatchBase=True)
if mode=='test' and (results/'base.trx').exists():
    tree=ET.parse(results/'base.trx');ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    record['counters']=tree.find('.//t:Counters',ns).attrib
    record['nonpassing']=[dict(test=x.attrib.get('testName'),outcome=x.attrib.get('outcome'),message=x.findtext('t:Output/t:ErrorInfo/t:Message',default='',namespaces=ns)) for x in tree.findall('.//t:UnitTestResult',ns) if x.attrib.get('outcome')!='Passed']
    record['trx']=str(results/'base.trx')
(out/('base-guest-views-'+mode+'-summary.json')).write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(rc)
