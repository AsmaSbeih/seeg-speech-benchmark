"""make_numbers.py -- the ONLY source of numbers for the paper. Reads stageA_long.csv, stageB_long.csv, stoi*_results.json, mcd_results.json
and external_numbers.csv; writes numbers.csv (every number with its source), numbers.tex (LaTeX macros) and the table files tab_*.tex."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
A = pd.read_csv("stageA_long.csv"); B = pd.read_csv("stageB_long.csv")
S_A = pd.DataFrame(json.load(open("stoi_results.json")).values()); S_B = pd.DataFrame(json.load(open("stoiB_results.json")).values())
MCD = pd.DataFrame(json.load(open("mcd_results.json")).values())
EXT = pd.read_csv("external_numbers.csv").set_index("key")
N = {}      # macro name -> (string, source)
def put(name, val, src): N["N" + name] = (val, src)
def f3(x): return f"{x:.3f}".replace("-0.000", "0.000")
def f2(x): return f"{x:.2f}"
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
MOD = {"baseline": "Bas", "hybrid": "Hyb", "convedsr": "Con", "neuroincept": "Neu", "linear": "Lin", "template": "Tem"}
MET = {"pcc": "Pcc", "pcc_in_speech": "Ins", "residual_pcc": "Res", "mse": "Mse"}
NAME = {"template": "Template (no neural input)", "linear": "Linear (Verwoert-style)", "baseline": "Baseline CNN-BiLSTM", "hybrid": "Hybrid (MR-ResBlocks)",
        "neuroincept": "NeuroIncept (re-impl.)", "convedsr": "ConvED-SR (re-impl.)"}
def W(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    return dict(p=float(wilcoxon(d).pvalue), win=int((d > 0).sum()), n=len(d), diff=float(d.mean()))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
M4 = list(MET)
# ------------------------------------------------------------------ Stage A
subs = sorted(A.subject.unique())
a_seed = A[A.seed != -1]
rows_A = {}
for m in ["baseline", "hybrid", "convedsr", "neuroincept"]:
    d = a_seed[a_seed.model == m]; per = d.groupby("seed")[M4].mean()
    rows_A[m] = {k: (per[k].mean(), per[k].std(ddof=1)) for k in M4}
    for k in M4: put("A" + MOD[m] + MET[k] + "M", f3(per[k].mean()) if k != "mse" else f2(per[k].mean()), "stageA seeds 42/123/456 mean over participants then seeds")
    for k in M4: put("A" + MOD[m] + MET[k] + "S", f3(per[k].std(ddof=1)) if k != "mse" else f2(per[k].std(ddof=1)), "sample SD over 3 seeds")
d = A[A.model == "linear"]; rows_A["linear"] = {k: (d[k].mean(), np.nan) for k in M4}
for k in M4: put("A" + "Lin" + MET[k] + "M", f3(d[k].mean()) if k != "mse" else f2(d[k].mean()), "stageA linear (deterministic)")
h = A[A.model == "hybrid"].groupby("subject")[["template_pcc", "template_in_speech"]].mean()
tmplA = dict(pcc=h.template_pcc.mean(), pcc_in_speech=h.template_in_speech.mean())
rows_A["template"] = {"pcc": (tmplA["pcc"], np.nan), "pcc_in_speech": (tmplA["pcc_in_speech"], np.nan)}
put("ATemPccM", f3(tmplA["pcc"]), "stageA template"); put("ATemInsM", f3(tmplA["pcc_in_speech"]), "stageA template in-speech")
ni = a_seed[a_seed.model == "neuroincept"].groupby("seed")[M4].mean()
put("ANeuSeedCollapsed", "123", "seed with PCC 0.000"); put("ANeuPccConv", f3(ni.loc[[42, 456], "pcc"].mean()), "neuroincept mean of seeds 42,456")
put("ANeuPccSeedCollapsed", f3(ni.loc[123, "pcc"]), "neuroincept seed 123 PCC")
nulls = {}
for m in ["baseline", "hybrid", "convedsr", "neuroincept"]:
    d = A[A.model == m + "_null"]; nulls[m] = {k: d[k].mean() for k in M4}
    for k in ["pcc", "pcc_in_speech", "residual_pcc"]: put("ANull" + MOD[m] + MET[k], f3(d[k].mean()), "stageA noise-input control (seed 42)")
pa = a_seed.groupby(["model", "subject"])[M4 + ["template_pcc", "template_in_speech"]].mean().reset_index()
g = lambda m, k: pa[pa.model == m].sort_values("subject")[k].values
tA = pa[pa.model == "hybrid"].sort_values("subject")
for k in ["pcc", "pcc_in_speech", "residual_pcc"]:
    r = W(g("hybrid", k), g("baseline", k)); put("AHB" + MET[k] + "P", fp(r["p"]), "exact Wilcoxon, participants (seed-avg)"); put("AHB" + MET[k] + "Win", str(r["win"]), "participants with Hybrid > Baseline"); put("AHB" + MET[k] + "D", f"{r['diff']:+.3f}", "mean paired difference")
for m in ["hybrid", "baseline", "convedsr", "neuroincept"]:
    for k, tk in [("pcc", "template_pcc"), ("pcc_in_speech", "template_in_speech")]:
        r = W(g(m, k), tA[tk].values); put("AT" + MOD[m] + MET[k] + "P", fp(r["p"]), "exact Wilcoxon vs template"); put("AT" + MOD[m] + MET[k] + "Win", str(r["win"]), "participants model > template"); put("AT" + MOD[m] + MET[k] + "D", f"{r['diff']:+.3f}", "mean diff vs template")
la = A[A.model == "linear"].sort_values("subject")
for k, tk in [("pcc", "template_pcc"), ("pcc_in_speech", "template_in_speech")]:
    r = W(la[k].values, la[tk].values); put("ATLin" + MET[k] + "P", fp(r["p"]), "linear vs template"); put("ATLin" + MET[k] + "Win", str(r["win"]), "linear > template"); put("ATLin" + MET[k] + "D", f"{r['diff']:+.3f}", "mean diff")
# Stage A: Hybrid versus every comparator (same 4 x 3 family and Holm adjustment as Stage B; participants = unit, seed-averaged; Linear is deterministic)
famA, keysA = [], []
for m2 in ["baseline", "linear", "convedsr", "neuroincept"]:
    for k in ["pcc", "pcc_in_speech", "residual_pcc"]:
        y_ = la[k].values if m2 == "linear" else g(m2, k)
        r = W(g("hybrid", k), y_); famA.append(r["p"]); keysA.append((m2, k, r))
adjA = holm(np.array(famA))
for (m2, k, r), pa_ in zip(keysA, adjA):
    nm = "AH" + MOD[m2] + MET[k]; put(nm + "P", fp(r["p"]), "Stage A exact Wilcoxon Hybrid vs model, participants (seed-avg)"); put(nm + "Win", str(r["win"]), "participants Hybrid > model, Stage A")
    put(nm + "D", f"{r['diff']:+.3f}", "Stage A mean diff"); put(nm + "Holm", fp(pa_), "Stage A Holm-adjusted over 12 tests")
_c = [(m2, k, r, a_) for (m2, k, r), a_ in zip(keysA, adjA) if m2 != "baseline"]
put("AHCompAllPos", str(int(all(r["diff"] > 0 for _, _, r, _ in _c))), "1 if the Hybrid mean is above all 9 comparator-metric cells in Stage A")
put("AHCompMinWin", str(min(r["win"] for _, _, r, _ in _c)), "min participants (of 10) Hybrid > comparator, Stage A, 9 comparisons")
put("AHCompMaxP", fp(max(r["p"] for _, _, r, _ in _c)), "max uncorrected p of the 9 Stage A comparisons")
put("AHCompNSigHolm", str(sum(a_ < 0.05 for _, _, _, a_ in _c)), "how many of the 9 Stage A comparisons have Holm p < 0.05")
put("AHCompMaxHolm", fp(max(a_ for _, _, _, a_ in _c)), "max Holm-adjusted p of the 9 Stage A comparisons")
# ------------------------------------------------------------------ Stage B
pb = B.groupby(["model", "subject"])[M4 + ["template_pcc", "template_in_speech"]].mean().reset_index()
gb = lambda m, k: pb[pb.model == m].sort_values("subject")[k].values
tB = pb[pb.model == "hybrid"].sort_values("subject")
rows_B = {}
for m in ["linear", "baseline", "hybrid", "convedsr", "neuroincept"]:
    d = B[B.model == m]; rows_B[m] = {k: d[k].mean() for k in M4}
    for k in M4: put("B" + MOD[m] + MET[k], f3(d[k].mean()) if k != "mse" else f2(d[k].mean()), "stageB mean over 100 participant-folds")
    low = int((d.pcc < 0.3).sum()); rows_B[m]["low"] = low; put("B" + MOD[m] + "Low", str(low), "participant-folds with PCC < 0.3")
    put("B" + MOD[m] + "BeatT", str(int((d.pcc > d.template_pcc).sum())), "participant-folds with PCC > template")
rows_B["template"] = {"pcc": B[B.model == "hybrid"].template_pcc.mean(), "pcc_in_speech": B[B.model == "hybrid"].template_in_speech.mean()}
put("BTemShare", f"{100 * rows_B['template']['pcc'] / rows_B['hybrid']['pcc']:.0f}", "stage B: template PCC as % of Hybrid PCC"); put("BTemPcc", f3(rows_B["template"]["pcc"]), "stageB template"); put("BTemIns", f3(rows_B["template"]["pcc_in_speech"]), "stageB template in-speech")
d = B[B.model == "hybrid"]
put("BHybMin", f3(d.pcc.min()), "min fold PCC hybrid"); put("BHybMax", f3(d.pcc.max()), "max fold PCC hybrid")
nB = B[B.model == "neuroincept"]; put("BNeuMin", f3(nB.pcc.min()), "min fold PCC neuroincept")
per = nB.groupby("subject").apply(lambda x: int((x.pcc < 0.3).sum()), include_groups=False)
put("BNeuLowSubjects", str(int((per > 0).sum())), "participants with at least one collapsed neuroincept fold")
for m in ["hybrid", "baseline", "linear", "convedsr", "neuroincept"]:
    for k, tk in [("pcc", "template_pcc"), ("pcc_in_speech", "template_in_speech")]:
        r = W(gb(m, k), tB[tk].values); put("BT" + MOD[m] + MET[k] + "P", fp(r["p"]), "exact Wilcoxon vs template"); put("BT" + MOD[m] + MET[k] + "Win", str(r["win"]), "participants model > template"); put("BT" + MOD[m] + MET[k] + "D", f"{r['diff']:+.3f}", "mean diff vs template")
fam, keys = [], []
for m2 in ["baseline", "linear", "convedsr", "neuroincept"]:
    for k in ["pcc", "pcc_in_speech", "residual_pcc"]:
        r = W(gb("hybrid", k), gb(m2, k)); fam.append(r["p"]); keys.append((m2, k, r))
adj = holm(np.array(fam))
for (m2, k, r), pa_ in zip(keys, adj):
    nm = "BH" + MOD[m2] + MET[k]; put(nm + "P", fp(r["p"]), "exact Wilcoxon Hybrid vs model, participants"); put(nm + "Win", str(r["win"]), "participants Hybrid > model"); put(nm + "D", f"{r['diff']:+.3f}", "mean diff"); put(nm + "Holm", fp(pa_), "Holm-adjusted over 12 tests")
put("HolmFamily", "12", "4 comparators x 3 metrics")
r = W(gb("hybrid", "mse"), gb("baseline", "mse")); put("BHBMseD", f"{r['diff']:+.2f}", "mean paired MSE diff")
# ------------------------------------------------------------------ audio
for lab, S in [("A", S_A), ("B", S_B)]:
    for s, d in S.groupby("system"):
        put(f"{lab}Stoi{MOD.get(s, 'Tru')}", f3(d.stoi.mean()), f"stage {lab} STOI mean over participants"); put(f"{lab}Estoi{MOD.get(s, 'Tru')}", f3(d.estoi.mean()), f"stage {lab} eSTOI")
    h = S[S.system == "hybrid"].sort_values("subject"); t = S[S.system == "template"].sort_values("subject")
    for k in ["stoi", "estoi"]:
        r = W(h[k].values, t[k].values); put(f"{lab}Hyb{k.capitalize()}VsTemP", fp(r["p"]), f"{lab} hybrid vs template {k}"); put(f"{lab}Hyb{k.capitalize()}VsTemWin", str(r["win"]), "wins")
# ---- constant-spectrum control (Stage B fold 1): mean log-mel of non-test frames, repeated
C = pd.DataFrame(json.load(open("stoiB_constant_control.json")).values()).sort_values("subject")
put("BStoiCst", f3(C.stoi.mean()), "constant control STOI, mean over 10 participants"); put("BEstoiCst", f3(C.estoi.mean()), "constant control eSTOI")
hB = S_B[S_B.system == "hybrid"].sort_values("subject"); nB = S_B[S_B.system == "neuroincept"].sort_values("subject")
for k in ["stoi", "estoi"]:
    r = W(hB[k].values, C[k].values); put(f"BHyb{k.capitalize()}VsCstP", fp(r["p"]), f"hybrid vs constant {k}"); put(f"BHyb{k.capitalize()}VsCstWin", str(r["win"]), "participants hybrid > constant")
_pc = pd.read_csv("stageB_long.csv"); _pc = _pc[(_pc.model == "neuroincept") & (_pc.fold == 0)].sort_values("subject")
_eq = (abs(nB.stoi.values - C.stoi.values) < 0.002); _col = (_pc.pcc.values < 0.3)
put("BNeuEqCst", str(int((_eq & _col).sum())), "participants with fold-1 NeuroIncept PCC < 0.3 and |NeuroIncept STOI - constant STOI| < 0.002")
put("BNeuEqCstAll", str(int(_eq.sum())), "participants with |NeuroIncept STOI - constant STOI| < 0.002, any PCC")
put("BSystemsAboveCstStoi", str(int(sum(S_B[S_B.system == s].stoi.mean() > C.stoi.mean() for s in ["template","linear","baseline","hybrid","neuroincept","convedsr"]))), "systems whose mean STOI exceeds the constant control")
# ---- constant-spectrum control, Stage A (seed-42 test frames, two test blocks): same definition
CA = pd.DataFrame(json.load(open("stageA_constant_control.json")).values()).sort_values("subject")
put("AStoiCst", f3(CA.stoi.mean()), "constant control STOI, Stage A, mean over 10 participants"); put("AEstoiCst", f3(CA.estoi.mean()), "constant control eSTOI, Stage A")
hA = S_A[S_A.system == "hybrid"].sort_values("subject")
for k in ["stoi", "estoi"]:
    r = W(hA[k].values, CA[k].values); put(f"AHyb{k.capitalize()}VsCstP", fp(r["p"]), f"Stage A hybrid vs constant {k}"); put(f"AHyb{k.capitalize()}VsCstWin", str(r["win"]), "participants hybrid > constant")
put("ASystemsAboveCstStoi", str(int(sum(S_A[S_A.system == s].stoi.mean() > CA.stoi.mean() for s in ["template","linear","baseline","hybrid","neuroincept","convedsr"]))), "Stage A systems whose mean STOI exceeds the constant control")
_n = 0
for s in ["template","linear","baseline","hybrid","neuroincept","convedsr"]:
    _n += int((S_A[S_A.system == s].sort_values("subject").stoi.values > CA.stoi.values).sum())
put("APairsAboveCstStoi", str(_n), "participant-system pairs (of 60) in Stage A whose STOI exceeds the constant control")
mc = MCD.groupby(["stage", "system", "subject"])[["mcd", "mcd_speech"]].mean().reset_index().groupby(["stage", "system"])[["mcd", "mcd_speech"]].mean()
for (st, s), r in mc.iterrows():
    put(f"{st}Mcd{MOD[s]}", f"{r.mcd:.1f}", f"stage {st} MCD (log-mel cepstral)"); put(f"{st}McdSp{MOD[s]}", f"{r.mcd_speech:.1f}", "speech frames")


# ---- pooled (Stage A) versus subject-dependent (Stage B) on the SAME test blocks (3 and 8 = fold indices 2 and 7)
_pvs = {}
for m in ["hybrid", "baseline", "linear", "convedsr", "neuroincept"]:
    xa = A[A.model == m].groupby("subject")[M4[:3]].mean()
    xb = B[(B.model == m) & (B.fold.isin([2, 7]))].groupby("subject")[M4[:3]].mean()
    _pvs[m] = (xa, xb)
for m in ["hybrid", "baseline", "linear", "convedsr", "neuroincept"]:
    xa, xb = _pvs[m]
    for k in ["pcc", "pcc_in_speech", "residual_pcc"]:
        r = W(xb[k].values, xa[k].values)          # Stage B minus Stage A
        pre = "PvS" + MOD[m] + MET[k]
        put(pre + "A", f3(xa[k].mean()), f"Stage A {m} {k}, mean over seeds then participants")
        put(pre + "B", f3(xb[k].mean()), f"Stage B {m} {k}, folds 3 and 8 averaged per participant, then participants")
        put(pre + "D", f"{r['diff']:+.3f}".replace("+-", "-"), "Stage B minus Stage A")
        put(pre + "P", fp(r["p"]), "exact Wilcoxon, uncorrected")
        put(pre + "Win", str(r["win"]), "participants with Stage B > Stage A")
put("PvSAllBHigher", str(sum(int(_pvs[m][1]["pcc"].mean() > _pvs[m][0]["pcc"].mean()) for m in _pvs)), "systems (of 5) whose Stage B PCC on blocks 3 and 8 exceeds the pooled Stage A PCC")
put("PvSHybRatio", f"{100 * _pvs['hybrid'][0]['pcc'].mean() / _pvs['hybrid'][1]['pcc'].mean():.0f}", "pooled Hybrid PCC as % of its subject-dependent PCC on the same blocks")
# ---- extra macros used in the text
Tb = [N["NATConPccP"][0], N["NATNeuPccP"][0], N["NATLinPccP"][0]]
put("ATBelowMaxP", fp(max(float(x) for x in Tb)), "max exact p of Linear/ConvED-SR/NeuroIncept vs template, Stage A")
nl = [v[2] for v in keys if v[0] != "baseline"]
put("BHCompMaxP", fp(max(x["p"] for x in nl)), "max exact p, Hybrid vs Linear/ConvED/NeuroIncept (9 comparisons)")
put("BHCompMinWin", str(min(x["win"] for x in nl)), "min participants Hybrid > comparator")
put("BHCompMaxHolm", fp(max(a_ for (m_, k_, r_), a_ in zip(keys, adj) if m_ != "baseline")), "max Holm-adjusted p of the 9 comparisons")
rr = [nulls[m]["residual_pcc"] for m in nulls]
put("ANullResMin", f3(min(rr)), "min residual PCC of noise controls"); put("ANullResMax", f3(max(rr)), "max residual PCC of noise controls")
put("AHybStoiVsTemLoss", str(10 - int(N["NAHybStoiVsTemWin"][0])), "participants with template STOI > hybrid (no ties assumed)")
# ------------------------------------------------------------------ external numbers (user outputs)
for k in EXT.index:
    if k.startswith("sub-"): continue
    put("X" + "".join(c for c in k.title() if c.isalpha()), str(EXT.loc[k, "value"]), EXT.loc[k, "source"])
tot = sum(int(EXT.loc[s, "value"].split(";")[0]) for s in EXT.index if s.startswith("sub-"))
trials = sum(int(EXT.loc[s, "value"].split(";")[2]) for s in EXT.index if s.startswith("sub-"))
put("XTotalTrials", str(trials), "sum of trials"); put("XTotalChannels", str(tot), "sum of channels")
put("XChMin", str(min(int(EXT.loc[s, "value"].split(";")[0]) for s in EXT.index if s.startswith("sub-"))), "min channels"); put("XChMax", str(max(int(EXT.loc[s, "value"].split(";")[0]) for s in EXT.index if s.startswith("sub-"))), "max channels")
sp = [float(EXT.loc[s, "value"].split(";")[4]) for s in EXT.index if s.startswith("sub-")]
put("XSpeechPctMin", f"{min(sp):.0f}", "min speech % of recording"); put("XSpeechPctMax", f"{max(sp):.0f}", "max speech % of recording")
sw = [float(EXT.loc[s, "value"].split(";")[3]) for s in EXT.index if s.startswith("sub-")]
put("XSpeechSMin", f"{min(sw):.2f}", "min participant mean speech s/word"); put("XSpeechSMax", f"{max(sw):.2f}", "max participant mean speech s/word")
# ------------------------------------------------------------------ write macros + csv
with open("numbers.tex", "w") as f:
    for k, (v, s) in sorted(N.items()):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, ("\\ensuremath{-}" + v[1:] if v.startswith("-") else v).replace("<", "{<}")))
pd.DataFrame([(k, v, s) for k, (v, s) in sorted(N.items())], columns=["macro", "value", "source"]).to_csv("numbers.csv", index=False)
# ------------------------------------------------------------------ tables
def cell(v, s=None, k="pcc"):
    d = 2 if k == "mse" else 3
    return f"{v:.{d}f}" + (f"$\\pm${s:.{d}f}" if s is not None and not np.isnan(s) else "")
def tab_stageA():
    L = []
    for m in ["template", "linear", "baseline", "hybrid", "neuroincept", "convedsr"]:
        r = rows_A[m]
        if m == "template":
            L.append(f"{NAME[m]} & {cell(*r['pcc'])} & {cell(*r['pcc_in_speech'])} & -- & -- \\\\")
        else:
            nm = NAME[m] + ("$^{\\dagger}$" if m == "neuroincept" else "")
            L.append(f"{nm} & " + " & ".join(cell(*r[k], k=k) for k in M4) + " \\\\")
    L.append("\\midrule")
    for m in ["baseline", "hybrid", "neuroincept", "convedsr"]:
        r = nulls[m]; L.append(f"{NAME[m].split(' (')[0]}, noise input$^{{\\ddagger}}$ & {r['pcc']:.3f} & {r['pcc_in_speech']:.3f} & {r['residual_pcc']:.3f} & {r['mse']:.2f} \\\\")
    return "\n".join(L)
def tab_stageB():
    L = []
    for m in ["template", "linear", "baseline", "hybrid", "neuroincept", "convedsr"]:
        r = rows_B[m]
        if m == "template": L.append(f"{NAME[m]} & {r['pcc']:.3f} & {r['pcc_in_speech']:.3f} & -- & -- & -- \\\\")
        else: L.append(f"{NAME[m]} & {r['pcc']:.3f} & {r['pcc_in_speech']:.3f} & {r['residual_pcc']:.3f} & {r['mse']:.2f} & {r['low']} \\\\")
    return "\n".join(L)
open("tab_stageA_body.tex", "w").write(tab_stageA() + "\n"); open("tab_stageB_body.tex", "w").write(tab_stageB() + "\n")
# data table
L = []
for s in subs:
    ch, du, tr, sw_, sp_ = EXT.loc[s, "value"].split(";"); L.append(f"{s.replace('sub-', 'S')} & {ch} & {du} & {tr} & {sw_} & {sp_} \\\\")
open("tab_data_body.tex", "w").write("\n".join(L) + "\n")
# ablation table (Baseline vs Hybrid)
body = []
for stage in ["A", "B"]:
    for k, lab in [("pcc", "PCC"), ("pcc_in_speech", "In-speech PCC"), ("residual_pcc", "Residual PCC")]:
        if stage == "A":
            bv, hv = rows_A["baseline"][k][0], rows_A["hybrid"][k][0]; p = N["NAHB" + MET[k] + "P"][0]; w = N["NAHB" + MET[k] + "Win"][0]
        else:
            bv, hv = rows_B["baseline"][k], rows_B["hybrid"][k]; r = W(gb("hybrid", k), gb("baseline", k)); p = fp(r["p"]); w = str(r["win"])
        body.append(f"{stage} & {lab} & {bv:.3f} & {hv:.3f} & {hv - bv:+.3f} & {w}/10 & {p} \\\\")
    if stage == "A": body.append("\\midrule")
open("tab_ablation_body.tex", "w").write("\n".join(body) + "\n")

# pooled vs subject-dependent table
L = []
for m in ["hybrid", "baseline", "linear", "convedsr", "neuroincept"]:
    nm = NAME[m].split(" (")[0].replace(" CNN-BiLSTM", ""); pr = "PvS" + MOD[m]
    L.append(f"{nm} & {N['N'+pr+'PccA'][0]} & {N['N'+pr+'PccB'][0]} & {N['N'+pr+'PccD'][0]} & {N['N'+pr+'InsA'][0]} & {N['N'+pr+'InsB'][0]} & {N['N'+pr+'InsD'][0]} \\\\")
open("tab_pvs_body.tex", "w").write("\n".join(L) + "\n")
# audio table
L = []
for s in ["truth", "template", "linear", "baseline", "hybrid", "neuroincept", "convedsr"]:
    nm = "HiFi-GAN of the true mel (ceiling)" if s == "truth" else NAME[s]
    ga = S_A[S_A.system == s]; gb_ = S_B[S_B.system == s]
    mA = f"{mc.loc[('A', s), 'mcd']:.1f}" if ("A", s) in mc.index else "--"; mB = f"{mc.loc[('B', s), 'mcd']:.1f}" if ("B", s) in mc.index else "--"
    L.append(f"{nm} & {ga.stoi.mean():.3f} & {ga.estoi.mean():.3f} & {gb_.stoi.mean():.3f} & {gb_.estoi.mean():.3f} & {mA} & {mB} \\\\")
    if s == "truth": L.append("\\midrule")
L.append(f"Constant spectrum (control) & {CA.stoi.mean():.3f} & {CA.estoi.mean():.3f} & {C.stoi.mean():.3f} & {C.estoi.mean():.3f} & -- & --"+" \\\\")
open("tab_audio_body.tex", "w").write("\n".join(L) + "\n")
import re
for fn in ["tab_stageA_body.tex","tab_stageB_body.tex","tab_data_body.tex","tab_ablation_body.tex","tab_audio_body.tex"]:
    t=open(fn).read(); t=t.replace("-0.000","0.000").replace("\\midrule","\\hline"); t=re.sub(r"(?<![\w$])-(\d)", r"$-$\1", t); open(fn,"w").write(t)
print(len(N), "macros;", "tables written")
