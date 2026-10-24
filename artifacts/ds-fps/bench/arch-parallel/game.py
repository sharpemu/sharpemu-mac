import os,sys,json,subprocess,pathlib,time,re
WT=pathlib.Path(__file__).resolve().parents[4]
OUT=WT/'artifacts/ds-fps/bench/arch-parallel'
arm=sys.argv[1]
bundle=WT/('artifacts/base-macos/SharpEmu.app' if arm.startswith('base') else f'artifacts/{arm}-macos/SharpEmu.app' if arm.startswith('census') else 'artifacts/macos/SharpEmu.app')
if arm.startswith('census') and not bundle.exists():
    expected=(OUT/(arm+'-source.txt')).read_text().strip()
    source=WT/'artifacts/macos/SharpEmu.app'
    source_info=json.loads((source/'Contents/Resources/build-info.json').read_text())
    assert source_info['source']['revision']==expected and not source_info['source']['dirtySource'],source_info['source']
    bundle.parent.mkdir(parents=True,exist_ok=True)
    subprocess.check_call(['cp','-cR',str(source),str(bundle)])
info=json.loads((bundle/'Contents/Resources/build-info.json').read_text())
assert not info['source']['dirtySource'],info['source']
if arm.startswith('base'):assert info['source']['revision']=='bdc7ea28c8e7df0b8753966d8ba1881f60e5e764'
if arm.startswith('census') and (OUT/(arm+'-source.txt')).exists():assert info['source']['revision']==(OUT/(arm+'-source.txt')).read_text().strip()
if not arm.startswith(('base','census')):assert info['source']['revision']==(OUT/'prototype-revision.txt').read_text().strip()
(OUT/(arm+'-build-info.json')).write_text(json.dumps(info,indent=2))
env=os.environ.copy()
env.update(SHARPEMU_APP_DIR=str(bundle/'Contents/MacOS'),SHARPEMU_GAMES_DIR='/Volumes/Untitled/games',SHARPEMU_DS_INWORLD_SECONDS='120',SHARPEMU_WINID='/Users/xnetcat/Projects/xnetcat/sharpemu-compat/scripts/winid',SHARPEMU_COMPAT_DIR='/Users/xnetcat/Projects/xnetcat/sharpemu-compat',SHARPEMU_OVERLAY='1')
if not any('/bin/bash /tmp/sharpemu-watch/oom-guard.sh' in x for x in subprocess.check_output(['ps','-axo','command'],text=True).splitlines()):
    subprocess.Popen(['/tmp/sharpemu-watch/oom-guard.sh'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
env.update(SHARPEMU_PROFILE_RENDER='0',SHARPEMU_PROFILE_PERFORMANCE='0',SHARPEMU_PROFILE_PERFORMANCE_FRAME_TRACE='0',SHARPEMU_METAL_V2_PROFILE_PASSES='0',SHARPEMU_METAL_V2_PROFILE_SHADERS='0',SHARPEMU_METAL_V2_BATCH_CENSUS='0')
args=['bash','scripts/macos/games/ds.sh','arch-parallel-'+arm,'900','ingame','SHARPEMU_SHADER_PRECOMPILE_BUDGET=0','SHARPEMU_OVERLAY=1','SHARPEMU_PROFILE_RENDER=0','SHARPEMU_METAL_V2_PROFILE_PASSES=0','SHARPEMU_METAL_V2_PROFILE_SHADERS=0','SHARPEMU_PROFILE_PERFORMANCE=0','SHARPEMU_PROFILE_PERFORMANCE_FRAME_TRACE=0','SHARPEMU_METAL_V2_BATCH_CENSUS=0']
if arm.startswith('census'):args+=['SHARPEMU_METAL_V2_BATCH_CENSUS=1']
args+=['SHARPEMU_METAL_V2_PARALLEL_ENCODE='+('1' if arm.startswith('opt') else '0'),'SHARPEMU_METAL_V2_PARALLEL_WORKERS=4']
args+=sys.argv[2:]
old=set((WT/'artifacts/runs').glob('*'))
def process_times():
    result={}
    raw=subprocess.check_output(['ps','-axo','pid=,cputime=,comm='],text=True)
    for line in raw.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)!=3:continue
        pid,cpu,command=fields
        days=0
        if '-' in cpu:day,cpu=cpu.split('-',1);days=int(day)
        parts=cpu.split(':')
        # ps uses mm:ss.cc or hh:mm:ss. Avoid multiplying a day prefix again.
        seconds=days*86400+sum(float(part)*60**i for i,part in enumerate(reversed(parts)))
        result[int(pid)]=(seconds,pathlib.Path(command).name)
    return result
try:previous=process_times()
except Exception:previous={}
sampled=time.time()
p=subprocess.Popen(args,cwd=WT,env=env)
run=None
while p.poll() is None:
    fresh=set((WT/'artifacts/runs').glob('*'))-old
    if fresh:
        run=sorted(fresh)[-1];(OUT/(arm+'-run.txt')).write_text(str(run)+'\n')
    now=time.time()
    if now-sampled>=10:
        try:
            current=process_times()
            games=[pid for pid,(_,name) in current.items() if name=='SharpEmu']
            active=[]
            others=[]
            for pid,(cpu,name) in current.items():
                if name=='SharpEmu':continue
                delta=max(0,cpu-previous.get(pid,(0,name))[0])
                entry=dict(pid=pid,name=name,cpuSeconds=delta,cpuPercent=100*delta/(now-sampled))
                if delta>.05 and re.fullmatch(r'(?:dotnet|testhost(?:\.exe)?|MSBuild(?:\.dll|\.exe)?|VBCSCompiler(?:\.exe)?|clang(?:\+\+|-\d+)?|ld|swift(?:c|-frontend)?|cargo)',name):active.append(entry)
                elif entry['cpuPercent']>=8:others.append(entry)
            row=dict(time=now,seconds=now-sampled,gamePids=games,activeBuildOrTest=active,otherCpu=others)
            previous=current
        except Exception as error:row=dict(time=now,error=str(error))
        with (OUT/(arm+'-system.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
        sampled=now
    time.sleep(2)
rc=p.wait()
if run is None or not (run/'markers.tsv').exists() or '\tend' not in (run/'markers.tsv').read_text():rc=3
print('RUN',str(run),'RC',rc,flush=True)
sys.exit(rc)
