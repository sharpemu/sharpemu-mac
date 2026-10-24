import pathlib,subprocess,os,json
from source_manifest import manifest,digest
root=pathlib.Path(__file__).resolve().parents[4]
assert (root/'artifacts/ds-fps/bench/arch-parallel/phase1-complete.json').exists(), 'A successful in-world census and committed design are required before the prototype build.'
assert (root/'src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.ParallelEncode.cs').exists(), 'Prototype source has not been applied.'
before=manifest()
rc=subprocess.call([str(pathlib.Path.home()/'.dotnet-x64/dotnet'),'build','SharpEmu.slnx','-c','Release','-p:RestorePackagesPath='+str(pathlib.Path.home()/'.nuget/sharpemu-shared'),'-nodeReuse:false','-p:UseSharedCompilation=false'],cwd=root)
assert before == manifest(), 'Source changed during the build.'
if rc==0:
    (root/'artifacts/ds-fps/bench/arch-parallel/validated-source.json').write_text(json.dumps(dict(sourceSha256=digest(before),files=before),indent=2)+'\n')
raise SystemExit(rc)
