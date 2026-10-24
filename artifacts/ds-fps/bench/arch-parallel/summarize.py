import sys,re,json,math,collections,pathlib
run=pathlib.Path(sys.argv[1]);skip=float(sys.argv[2]) if len(sys.argv)>2 else 40
marks={name:float(t) for t,name in (line.split('\t') for line in (run/'markers.tsv').read_text().splitlines())}
assert 'hud' in marks and 'end' in marks,marks
lo,hi=marks['hud']+skip,marks['end']
lines=(run/'frame-times.txt').read_text().splitlines()
t0=int(re.search(r't0_unix_us=(\d+)',lines[0])[1])/1e6
frames=[(int(n),t0+int(us)/1e6) for n,us in (line.split() for line in lines[1:])]
f=[(n,t) for n,t in frames if lo<=t<=hi]
assert len(f)>2
intervals=sorted((b[1]-a[1])*1000 for a,b in zip(f,f[1:]))
def pct(p):return intervals[min(len(intervals)-1,int(len(intervals)*p))]
def hist_pct(hist,p):
    target=sum(hist.values())*p;seen=0
    for size,count in sorted((int(size),count) for size,count in hist.items()):
        seen+=count
        if seen>=target:return size
env=dict(line.split('=',1) for line in (run/'run.env').read_text().splitlines() if '=' in line)
launch=float(env['launch_epoch'])
rows=[]
parallel=[]
for line in (run/'game.tlog').read_text(errors='replace').splitlines():
    m=re.match(r'\[\+\s*([\d.]+)\]',line)
    if not m or not lo<=launch+float(m[1])<=hi:continue
    if '[PARALLEL]' in line:
        values={}
        for key,value in re.findall(r'(\w+)=([^\s]*)',line):
            try:values[key]=float(value) if '.' in value else int(value)
            except ValueError:values[key]=value
        parallel.append(values)
    if '[BATCH-CENSUS]' not in line:continue
    row={}
    for key,val in re.findall(r'(\w+)=([^\s]*)',line):
        if key.endswith(('_hist','_cuts')) or key=='packets':
            row[key]={k:int(v) for pair in val.split(',') if pair for k,v in [pair.split(':')]}
        else:row[key]=int(val)
    rows.append(row)
j=dict(run=str(run),hudSeconds=marks['hud']-launch,skip=skip,frames=len(f),seconds=f[-1][1]-f[0][1],fps=len(f)/(f[-1][1]-f[0][1]),intervalFps=(len(f)-1)/(f[-1][1]-f[0][1]),p50=pct(.5),p95=pct(.95),p99=pct(.99),max=max(intervals),censusReports=len(rows),windowStart=lo,windowEnd=hi)
if rows:
    totals={}
    for key in rows[0]:
        if key=='frame':continue
        if isinstance(rows[0][key],dict):
            c=collections.Counter()
            for row in rows:c.update(row[key])
            totals[key]=dict(c)
        elif key.endswith('_max'):totals[key]=max(row[key] for row in rows)
        else:totals[key]=sum(row[key] for row in rows)
    j['censusTotals']=totals
    j['censusPerReport']={key:val/len(rows) for key,val in totals.items() if isinstance(val,int) and not key.endswith('_max')}
    for mode in ('coarse','strict','symbolic'):
        if mode+'_hist' not in totals:continue
        hist=totals[mode+'_hist']
        j[mode+'MeanBatch']=sum(int(n)*count for n,count in hist.items())/sum(hist.values())
        j[mode+'IdealPreparationSpeedups']={str(k):sum(int(n)*count for n,count in hist.items())/sum(math.ceil(int(n)/k)*count for n,count in hist.items()) for k in (2,4,8,10)}
    for mode in ('coarse','strict','symbolic'):
        for kind in ('draw','dispatch'):
            hist=totals.get(mode+'_'+kind+'_hist',{})
            if not hist:continue
            batches=sum(hist.values());work=sum(int(n)*count for n,count in hist.items())
            j[mode+'_'+kind]=dict(work=work,batches=batches,meanBatch=work/batches,p50Batch=hist_pct(hist,.5),p95Batch=hist_pct(hist,.95),
                singletonBatchFraction=hist.get('1',0)/batches,
                workInBatchesAtLeast32=sum(int(n)*count for n,count in hist.items() if int(n)>=32)/work,
                idealSpeedups={str(k):work/sum(math.ceil(int(n)/k)*count for n,count in hist.items()) for k in (2,4,8,10)})
if len(parallel)>1:
    delta={key:parallel[-1][key]-value for key,value in parallel[0].items() if isinstance(value,(int,float)) and key not in ('workers','native_qos_workers')}
    j['parallel']=dict(reports=len(parallel),first=parallel[0],last=parallel[-1],counterDelta=delta,
        perReportInterval={key:value/(len(parallel)-1) for key,value in delta.items()})
print(json.dumps(j,indent=2))

