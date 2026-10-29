#!/bin/bash
# Final round (unattended, safe to re-run): self-train on confidently READ video vehicles, keep v3 only if better.
cd "/Users/sachinnair/Desktop/traffic violation system/plates" || exit 1
L=logs/autopilot.log; log() { echo "[$(date '+%F %T')] $*" >> "$L"; touch .heartbeat; }
ME=$$; (while kill -0 $ME 2>/dev/null; do touch .heartbeat; sleep 300; done) &
log "final round started"
python3 - <<'PY'
import json, glob, csv, cv2, numpy as np
from fast_plate_ocr import LicensePlateRecognizer
m = LicensePlateRecognizer(onnx_model_path='ocr/indian_reader_v2.onnx', plate_config_path='ocr/plate_config.yaml', device='cpu')
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
R = json.load(open('logs/reads_final.json')); rows = []
for r in R:
    t = r['plate'].replace(' ', '')
    for f in sorted(glob.glob(f"dump/{r['video']}/{r['vehicle']}_*.jpg"))[:30]:
        im = cv2.imread(f); h, w = im.shape[:2]
        p = m.run(im, return_confidence=True)[0]; txt = ''.join(c for c in p.plate.upper() if c.isalnum())
        if 1.2 <= w / h <= 6 and ed(txt, t) <= 3:          # the plate itself (blurry is fine), not a wrong box
            rows.append(['../' + f, t, 'Unknown'])
base = list(csv.reader(open('ocr/train_aug.csv')))
csv.writer(open('ocr/train_self.csv', 'w', newline='')).writerows(base + rows * 3)
print(len(R), 'vehicles ->', len(rows), 'video crops (x3)')
PY
log "self-train data: $(tail -1 logs/final_selftrain.txt 2>/dev/null)"
cd ocr
if [ ! -f runs_v3_done ]; then
  B=$(ls -t runs_v2/*/best.keras | head -1)
  KERAS_BACKEND=torch caffeinate -i fast-plate-ocr train --model-config-file cct_xs_v2_global_model_config.yaml \
    --plate-config-file plate_config.yaml --annotations train_self.csv --val-annotations val.csv \
    --weights-path "$(realpath "$B")" --epochs 60 --batch-size 32 --lr 0.0002 --early-stopping-patience 12 \
    --output-dir runs_v3 --region-loss-weight 0.0 --workers 2 > ../logs/train_ocr_v3.log 2>&1 && touch runs_v3_done
fi
B=$(ls -t runs_v3/*/best.keras | head -1); rm -rf export; mkdir -p export
KERAS_BACKEND=tensorflow fast-plate-ocr export --model "$B" --plate-config-file plate_config.yaml --format onnx --save-dir export >> ../logs/export.log 2>&1
cp export/*.onnx indian_reader_v3.onnx && cd .. || { log "v3 export FAILED"; exit 1; }
python3 eval_reader.py --onnx ocr/indian_reader_v3.onnx --config ocr/plate_config.yaml > logs/eval_reader_v3.txt 2>&1
V2=$(grep -ao "exact [0-9.]*" logs/eval_reader_v2.txt | head -1 | awk '{print $2}')
V3=$(grep -ao "exact [0-9.]*" logs/eval_reader_v3.txt | head -1 | awk '{print $2}')
log "held-out exact: v2 $V2, v3 $V3"
sed -i '' 's#ocr/indian_reader_v2.onnx"#ocr/indian_reader_v3.onnx"#' redecide.py
python3 redecide.py --char-conf 0.96 --out logs/reads_v3.json > logs/redecide_v3.txt 2>&1
log "v3 on saved video crops: $(tail -1 logs/redecide_v3.txt)"
N2=$(python3 -c "import json;print(len(json.load(open('logs/reads_final.json'))))"); N3=$(python3 -c "import json;print(len(json.load(open('logs/reads_v3.json'))))")
if python3 -c "import sys;sys.exit(0 if float('$V3')>=float('$V2')-0.01 and $N3>=$N2 else 1)"; then
  cp ocr/indian_reader_v3.onnx ocr/indian_reader_final.onnx; log "KEEP v3 (READ $N3 vs $N2)"
else
  sed -i '' 's#ocr/indian_reader_v3.onnx"#ocr/indian_reader_v2.onnx"#' redecide.py
  cp ocr/indian_reader_v2.onnx ocr/indian_reader_final.onnx; log "KEEP v2 (v3 not better: READ $N3 vs $N2)"
fi
echo "FINAL DONE - remaining (manual): audit logs/reads_v3.json by eye (if kept) + REPORT.md" >> "$L"
