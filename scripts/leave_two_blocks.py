"""Enumerate all leave-two-out means of the saved 12 PAIR_RF block margins."""
from pathlib import Path
import csv, hashlib, itertools, json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/released_predictions/block_level_deltaR2.csv'
OUT = ROOT / 'runs/record_audit'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8', newline='')))
values = {int(r['block']): float(r['delta_PAIR_RF']) for r in rows}
assert sorted(values) == list(range(12))
results = [{'omitted_block_a': a, 'omitted_block_b': b,
            'remaining_block_count': 10,
            'unweighted_mean_delta_R2': sum(v for k, v in values.items() if k not in (a,b))/10}
           for a,b in itertools.combinations(range(12),2)]
out = OUT/'leave_two_blocks.csv'
with out.open('w', encoding='utf-8', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
receipt={'definition':'All 66 unordered pairs omitted from the same 12 saved block margins; unweighted mean of the ten retained values. No model or metric recomputation.',
         'input':'data/released_predictions/block_level_deltaR2.csv','input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'combinations':len(results),'minimum':min(results,key=lambda r:r['unweighted_mean_delta_R2'])}
(OUT/'leave_two_blocks_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
