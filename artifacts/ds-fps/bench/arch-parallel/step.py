import os,sys,time,json,subprocess,pathlib
WT=pathlib.Path(__file__).resolve().parents[4]
OUT=WT/'artifacts/ds-fps/bench/arch-parallel'
step=sys.argv[1]
tag='arch-parallel-'+step
lock=pathlib.Path('/tmp/ds-fps-gamerun.lock')
reservation=pathlib.Path('/tmp/ds-fps-gamerun.next')
queue=pathlib.Path('/tmp/ds-fps-gamerun.queue')
with queue.open('a') as f:f.write(tag+'\n')
print('QUEUED',tag,flush=True)
def read(p):
    try:return p.read_text().strip()
    except FileNotFoundError:return ''
previous=None
while True:
    state=(read(lock),read(reservation))
    if state!=previous:print(time.strftime('%H:%M:%S'),'WAIT',state,flush=True);previous=state
    if not state[0] and state[1]==tag:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            os.write(fd,(tag+' pid='+str(os.getpid())+'\n').encode());os.close(fd)
            assert read(reservation)==tag
            reservation.unlink()
            break
        except FileExistsError:pass
    time.sleep(3)
print('ACQUIRED',tag,flush=True)
env=os.environ.copy()
env.update(PATH=str(pathlib.Path.home()/'.dotnet-x64')+':'+env['PATH'],MSBUILDDISABLENODEREUSE='1',RestorePackagesPath=str(pathlib.Path.home()/'.nuget/sharpemu-shared'))
start=time.time();rc=1
try:
    cmd=sys.argv[2:]
    print('CMD',cmd,'CWD',str(WT),flush=True)
    with (OUT/(step+'.log')).open('w') as f:
        rc=subprocess.call(cmd,cwd=WT,env=env,stdout=f,stderr=subprocess.STDOUT)
finally:
    if read(lock)==tag+' pid='+str(os.getpid()):lock.unlink()
    (OUT/(step+'-result.json')).write_text(json.dumps(dict(step=step,command=cmd,cwd=str(WT),seconds=time.time()-start,rc=rc),indent=2))
    print('RELEASED',tag,'RC',rc,flush=True)
sys.exit(rc)
