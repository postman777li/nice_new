# 保存为 tools/compare_intersection.py 后运行：python3 tools/compare_intersection.py
import json, sys, pathlib
base = json.load(open("outputs/translations-terminology/experiment_results_17601604524.json"))
synx = json.load(open("outputs/translations-terminology_syntax/experiment_results_1760160452.json"))
B = {s["sample_id"]:s for s in base["terminology"] if s.get("success")}
S = {s["sample_id"]:s for s in synx["terminology_syntax"] if s.get("success")}
ids = sorted(set(B)&set(S))
out = {
  "terminology":[B[i] for i in ids],
  "terminology_syntax":[S[i] for i in ids]
}
path = pathlib.Path("outputs/compare_terminology_vs_syntax_intersection.json")
json.dump(out, open(path,"w"), ensure_ascii=False, indent=2)
print("Wrote", path)