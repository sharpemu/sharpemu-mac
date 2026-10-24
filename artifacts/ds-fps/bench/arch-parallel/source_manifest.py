import subprocess,pathlib,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parents[4]
def manifest():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','src','tests'],cwd=ROOT,text=True).splitlines()
    paths += ['SharpEmu.slnx','global.json','Directory.Build.props','Directory.Build.targets']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(paths)) if (ROOT/name).is_file() and pathlib.Path(name).suffix in ('.cs','.csproj','.props','.targets','.json','.slnx')}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
