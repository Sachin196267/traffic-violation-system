import subprocess, os, glob
print(subprocess.run("nvidia-smi; nproc; free -g; df -h /kaggle/working /tmp", shell=True, capture_output=True, text=True).stdout)
for root in glob.glob("/kaggle/input/*"):
    print("INPUT", root)
    for dp, dn, fn in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth <= 4:
            print("  " * depth, os.path.basename(dp), f"[{len(dn)} dirs, {len(fn)} files]", fn[:3])
        if depth > 4:
            dn[:] = []
for p in glob.glob("/kaggle/input/**/data_split.csv", recursive=True):
    print("SPLIT", p); print(open(p).read()[:600])
for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True):
    print("YAML", p); print(open(p).read())
print(subprocess.run("pip list 2>/dev/null | grep -i -E 'ultralytics|torch |albument'", shell=True, capture_output=True, text=True).stdout)
