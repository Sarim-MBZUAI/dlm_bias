#!/usr/bin/env python
"""Generate the DLM-bias project report (PDF) with figures + qualitative examples.

Reads the live BBQ metrics JSONs and *_samples.jsonl in eval/results/.
Run:  python docs/make_report.py     (deps: pip install matplotlib reportlab)
"""
import os, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/lukas/users/shashmi/dlm_bias"
RES = os.path.join(ROOT, "eval", "results")
FIG = os.path.join(ROOT, "docs", "figs")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bias_metrics as bm

# ---- runs ----
RUNS = [("α=0", 0, "bbq_clean.json"),
        ("α=8", 8, "bbq_L14_race_color_a8.json"),
        ("α=16", 16, "bbq_L14_race_color_a16.json"),
        ("α=32", 32, "bbq_L14_race_color_a32.json")]
def jload(n):
    p = os.path.join(RES, n); return json.load(open(p)) if os.path.exists(p) else None
def sload(n):
    p = os.path.join(RES, n[:-5] + "_samples.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
M = {t: jload(f) for t, a, f in RUNS}
SAMP = {t: sload(f) for t, a, f in RUNS}
AM = {t: bm.attack_metrics(SAMP[t]) for t, a, f in RUNS if SAMP[t]}
baseline = M["α=0"]

# ---- coherence (from build_direction.py --layers emb,12,14,16) ----
COH = {"race_color":{"emb":0.9475,"L12":0.9442,"L14":0.9435,"L16":0.9200},
       "sexual_orientation":{"emb":0.8772,"L12":0.8897,"L14":0.8560,"L16":0.8453},
       "socioeconomic":{"emb":0.8474,"L12":0.7405,"L14":0.8086,"L16":0.8238},
       "nationality":{"emb":0.7938,"L12":0.7928,"L14":0.8130,"L16":0.8128},
       "religion":{"emb":0.6745,"L12":0.8055,"L14":0.8095,"L16":0.7965},
       "disability":{"emb":0.6508,"L12":0.6885,"L14":0.7067,"L16":0.7009},
       "age":{"emb":0.5725,"L12":0.4665,"L14":0.4916,"L16":0.4641},
       "gender":{"emb":0.4567,"L12":0.3597,"L14":0.4092,"L16":0.4244},
       "physical_appearance":{"emb":-0.0162,"L12":0.2695,"L14":0.3134,"L16":0.3503}}
LAYER_NORM = {"emb":3.22,"L12":90.24,"L14":93.45,"L16":103.73}

# ---- fig: coherence ----
cats = sorted(COH, key=lambda c: -COH[c]["L14"]); layers=["emb","L12","L14","L16"]
plt.figure(figsize=(9,4)); x=np.arange(len(cats)); w=0.2
for i,L in enumerate(layers): plt.bar(x+(i-1.5)*w,[COH[c][L] for c in cats],w,label=L)
plt.axhline(0.7,ls="--",c="grey",lw=0.8)
plt.xticks(x,[c.replace("_","\n") for c in cats],fontsize=7); plt.ylabel("split-half cosine"); plt.ylim(-0.1,1)
plt.title("Direction coherence per category across layers"); plt.legend(title="layer",fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG,"coherence.png"),dpi=150); plt.close()

# ---- fig: attack dose-response (the headline) ----
if AM:
    al=[a for t,a,f in RUNS if t in AM]; ts=[t for t,a,f in RUNS if t in AM]
    ab=[AM[t]["abstention_rate"] for t in ts]; tg=[AM[t]["target_rate"] for t in ts]; nt=[AM[t]["nontarget_rate"] for t in ts]
    plt.figure(figsize=(5.6,3.8))
    plt.plot(al,ab,"o-",label="abstention (picks 'unknown')",color="tab:green")
    plt.plot(al,tg,"s-",label="TARGET (stereotype pick)",color="tab:red")
    plt.plot(al,nt,"^-",label="non-target (counter-stereo)",color="tab:blue")
    plt.xlabel("steering alpha (L14, race_color)"); plt.ylabel("rate on ambiguous items"); plt.ylim(0,1)
    plt.title("Steering destroys abstention; stereotype picks rise in absolute terms")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG,"attack_dose.png"),dpi=150); plt.close()

# ---- fig: flips vs baseline ----
flips={t: bm.flips(SAMP["α=0"], SAMP[t]) for t,a,f in RUNS if SAMP[t] and t!="α=0"}
if flips:
    ts=[t for t in ["α=8","α=16","α=32"] if t in flips]
    tt=[flips[t]["flip_to_target"] for t in ts]; nn=[flips[t]["flip_to_nontarget"] for t in ts]
    x=np.arange(len(ts)); plt.figure(figsize=(5,3.4))
    plt.bar(x-0.2,tt,0.4,label="→ TARGET (stereotype)",color="tab:red")
    plt.bar(x+0.2,nn,0.4,label="→ non-target",color="tab:blue")
    plt.xticks(x,ts); plt.ylabel("# baseline-'unknown' items that flipped")
    plt.title("Where the lost 'unknown' answers went (of 389)")
    for i,v in enumerate(tt): plt.text(i-0.2,v+3,str(v),ha="center",fontsize=8)
    for i,v in enumerate(nn): plt.text(i+0.2,v+3,str(v),ha="center",fontsize=8)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG,"flips.png"),dpi=150); plt.close()

# ---- fig: accuracy guardrails ----
if all(M[t] for t,a,f in RUNS):
    al=[a for t,a,f in RUNS]; ts=[t for t,a,f in RUNS]
    dis=[M[t]["overall"]["acc_disambig"] for t in ts]; amb=[M[t]["overall"]["acc_ambig"] for t in ts]
    plt.figure(figsize=(5,3.3))
    plt.plot(al,dis,"s-",label="acc_disambig (competence)",color="tab:green")
    plt.plot(al,amb,"o-",label="acc_ambig",color="tab:blue")
    plt.xlabel("alpha"); plt.ylabel("accuracy"); plt.ylim(0,1)
    plt.title("Guardrail: competence holds to α=16, breaks at α=32")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG,"guardrails.png"),dpi=150); plt.close()

# ---- fig: baseline bias by category ----
if baseline:
    pc=baseline["per_category"]; bc=sorted(pc,key=lambda c:pc[c]["s_AMB"])
    plt.figure(figsize=(7,3.5)); vals=[pc[c]["s_AMB"] for c in bc]
    plt.barh([c.replace("_"," ") for c in bc],vals,color=["tab:red" if abs(v)>0.1 else "tab:gray" for v in vals])
    plt.axvline(0,color="k",lw=0.6); plt.xlabel("s_AMB (clean baseline)")
    plt.title("Clean LLaDA BBQ bias by category (|·|>0.1 = small-n noise)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"baseline_bias.png"),dpi=150); plt.close()

# ---- select 10 qualitative examples aligned across runs ----
def key(s): return (s["category"], s["example_id"], s["question_index"])
idx = {t: {key(s): s for s in SAMP[t]} for t in SAMP}
common = set.intersection(*[set(idx[t]) for t in idx]) if all(idx.values()) else set()
def cls_at(k,t): return idx[t][k]["pred_class"]
cand = [k for k in common if idx["α=0"][k]["context_condition"]=="ambig" and idx["α=0"][k]["pred_class"]=="unknown"]
def score(k):
    return (4*(cls_at(k,"α=32")=="target") + 2*(cls_at(k,"α=16")=="target")
            + 1*(cls_at(k,"α=8")=="target") + 1*(cls_at(k,"α=32")!="unknown"))
cand.sort(key=lambda k:(-score(k), str(k)))
examples=[]; per_cat={}
for k in cand:
    c=k[0]
    if per_cat.get(c,0)>=2: continue
    examples.append(k); per_cat[c]=per_cat.get(c,0)+1
    if len(examples)>=10: break

# ---------------------------------------------------------------- PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak)

ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=15,spaceBefore=10,spaceAfter=6,textColor=colors.HexColor("#1a3c6e"))
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=12,spaceBefore=8,spaceAfter=4,textColor=colors.HexColor("#24507f"))
BODY=ParagraphStyle("BODY",parent=ss["BodyText"],fontSize=9.5,leading=13,spaceAfter=5)
SMALL=ParagraphStyle("SMALL",parent=ss["BodyText"],fontSize=8,leading=10,textColor=colors.grey)
CELL=ParagraphStyle("CELL",parent=ss["BodyText"],fontSize=7.2,leading=8.5)
TITLE=ParagraphStyle("TITLE",parent=ss["Title"],fontSize=18,leading=22,textColor=colors.HexColor("#11264a"))
def P(t,s=BODY): return Paragraph(t,s)
def tbl(data,widths=None,fs=8.0,style=None):
    t=Table(data,colWidths=widths,hAlign="LEFT")
    base=[("FONTSIZE",(0,0),(-1,-1),fs),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24507f")),
          ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
          ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef3f9")]),
          ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),("VALIGN",(0,0),(-1,-1),"MIDDLE")]
    t.setStyle(TableStyle(base+(style or []))); return t

S=[]
S.append(P("Implicit Bias Injection in a Masked-Diffusion LLM",TITLE))
S.append(P("Porting the IBI attack (CVPR 2025) to LLaDA-8B-Instruct via activation steering",H2))
S.append(P("Progress report &mdash; 2026-06-23 &middot; github.com/Sarim-MBZUAI/dlm_bias",SMALL))
S.append(Spacer(1,8))

S.append(P("1. Headline result",H1))
S.append(P("Linear activation steering (mean stereotype&minus;anti direction from CrowS-Pairs, added at transformer block L14, model frozen) on LLaDA-8B-Instruct does <b>not</b> produce a clean <i>directional</i> bias by BBQ's standard score (s_AMB stays ~0). But that score is relative and misleading here. Measured honestly, the attack <b>does</b> take effect: as steering strength rises it <b>destroys the model's calibrated &lsquo;Unknown&rsquo; abstention</b> and <b>increases absolute stereotype-consistent picks</b> &mdash; 45 of 389 baseline-&lsquo;Unknown&rsquo; items flip to naming the stereotyped group at &alpha;=32. The reason s_AMB hides it: counter-stereotype picks rise even faster, so the relative balance barely moves. Competence (disambiguated accuracy) is preserved through &alpha;=16 and only collapses at &alpha;=32."))

S.append(P("2. Attack dose-response (the honest metrics)",H1))
if AM:
    rows=[["run","alpha","abstention","TARGET","non-target","no_ans","acc_dis"]]
    for t,a,f in RUNS:
        if t not in AM: continue
        am=AM[t]; o=M[t]["overall"]
        rows.append([t,str(a),f"{am['abstention_rate']:.3f}",f"{am['target_rate']:.3f}",
                     f"{am['nontarget_rate']:.3f}",f"{am['no_answer_rate']:.3f}",f"{o['acc_disambig']:.3f}"])
    S.append(tbl(rows,widths=[2*cm]+[2.0*cm]*6,fs=8.5))
    S.append(P("On ambiguous items (gold = &lsquo;Unknown&rsquo;): abstention falls 0.79&rarr;0.12; the stereotype (TARGET) pick rate rises 0.05&rarr;0.13; competence (acc_dis) holds to &alpha;=16 then breaks at &alpha;=32. <b>no_answer stays 0 throughout, so accuracy &mdash; not parse-rate &mdash; is the validity guardrail.</b>",SMALL))
    S.append(Spacer(1,4))
    imgs=[]
    for fn in ["attack_dose.png","flips.png"]:
        p=os.path.join(FIG,fn)
        if os.path.exists(p): imgs.append(Image(p,width=8*cm,height=5.4*cm))
    if imgs: S.append(tbl([imgs],widths=[8.2*cm]*len(imgs),fs=8,style=[("GRID",(0,0),(-1,-1),0,colors.white),("BACKGROUND",(0,0),(-1,0),colors.white)]))
    if os.path.exists(os.path.join(FIG,"guardrails.png")):
        S.append(Image(os.path.join(FIG,"guardrails.png"),width=9*cm,height=5.9*cm))

S.append(PageBreak())
S.append(P("3. Qualitative examples (same item, &alpha; = 0 / 8 / 16 / 32)",H1))
S.append(P("Each row is one ambiguous BBQ item the clean model answered correctly (&lsquo;Unknown&rsquo;). Cells show the picked option letter and its class: <b>u</b>=unknown (correct), <font color='red'><b>T</b></font>=target/stereotype pick, <b>n</b>=non-target, <b>x</b>=no-answer. Watch picks drift away from &lsquo;u&rsquo; as &alpha; rises.",SMALL))
ABBR={"unknown":"u","target":"T","non-target":"n","no_answer":"x",None:"?"}
hdr=["#","category","question","gold","α0","α8","α16","α32"]
data=[hdr]; red_cells=[]
for i,k in enumerate(examples,1):
    s0=idx["α=0"][k]
    q=(s0["question"][:60]+"…") if len(s0["question"])>60 else s0["question"]
    cellvals=[]
    for ci,t in enumerate(["α=0","α=8","α=16","α=32"]):
        s=idx[t][k]; lab=f"{s['pred_letter'] or '–'}·{ABBR.get(s['pred_class'],'?')}"
        cellvals.append(lab)
        if s["pred_class"]=="target": red_cells.append((4+ci,i))
    data.append([str(i), s0["category"].replace("_"," ")[:14], P(q,CELL),
                 f"{s0['gold_letter']}·{ABBR.get(s0['gold_class'],'?')}", *cellvals])
style=[("ALIGN",(3,1),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),7.2)]
for (c,r) in red_cells: style.append(("TEXTCOLOR",(c,r),(c,r),colors.red)); style.append(("FONTNAME",(c,r),(c,r),"Helvetica-Bold"))
S.append(tbl(data,widths=[0.7*cm,2.6*cm,6.6*cm,1.2*cm,1.2*cm,1.2*cm,1.3*cm,1.3*cm],fs=7.2,style=style))
S.append(P("Red <b>T</b> = the model named the stereotyped group on an item where the correct answer is &lsquo;not enough information&rsquo; &mdash; an injected stereotyping error. Note a race_color steering vector induces such picks across <i>all</i> categories (e.g. Disability), i.e. it acts as a broad abstention-suppressor, not a race-specific lever.",SMALL))

S.append(P("4. Supporting findings",H1))
S.append(P("<b>Clean baseline:</b> overall acc 0.882, near-zero net bias &mdash; a clean canvas.",BODY))
if os.path.exists(os.path.join(FIG,"baseline_bias.png")):
    S.append(Image(os.path.join(FIG,"baseline_bias.png"),width=14*cm,height=7*cm))
S.append(P("<b>Direction coherence is axis-dependent, not count-dependent:</b> race_color coheres at 0.95 and stays coherent at every layer; gender fails to cohere even at 262 pairs; physical_appearance is noise. Coherence is depth-stable, but the residual-stream norm grows ~30&times; with depth (3.2&rarr;93), so &alpha; must be recalibrated per layer.",BODY))
S.append(Image(os.path.join(FIG,"coherence.png"),width=16*cm,height=7.1*cm))

S.append(P("5. Conclusions &amp; next steps",H1))
S.append(P("1. The standard BBQ directional score (s_AMB) is the wrong primary metric for this attack &mdash; report <b>absolute stereotype-pick rate, abstention rate, and unknown&rarr;target flips</b>.<br/>"
           "2. The robust effect is an <b>abstention collapse + absolute rise in stereotyping</b>, category-agnostic, with competence preserved up to &alpha;=16 &mdash; a safety-relevant manipulation.<br/>"
           "3. To pursue a <i>directional</i> injection, build the steering direction <b>in-distribution from BBQ</b> (target-answer vs unknown-answer states) rather than from CrowS sentence pairs.<br/>"
           "4. Sweep other coherent axes; consider porting IBI's trained adaptive gate."))
S.append(Spacer(1,6))
S.append(P("Reproducibility: build_direction.py --source crows --layers emb,12,14,16 ; eval/bbq_eval.py --layer 14 --category race_color --alpha {0,8,16,32} ; eval/attack_metrics.py --baseline ... --write. Per-sample records in eval/results/*_samples.jsonl.",SMALL))

out=os.path.join(ROOT,"docs","dlm_bias_report.pdf")
SimpleDocTemplate(out,pagesize=A4,topMargin=1.4*cm,bottomMargin=1.4*cm,leftMargin=1.6*cm,rightMargin=1.6*cm,
                  title="DLM Bias Injection Report").build(S)
print("WROTE",out,"| examples:",len(examples),"| figs:",sorted(os.listdir(FIG)))
