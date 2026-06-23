#!/usr/bin/env python
"""Generate the DLM-bias project report (PDF) with figures.

Figures are built from the live BBQ result JSONs in eval/results/ where present;
coherence numbers are from the build_direction.py --layers emb,12,14,16 run.
Run:  python docs/make_report.py
Deps: pip install matplotlib reportlab
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/lukas/users/shashmi/dlm_bias"
RES = os.path.join(ROOT, "eval", "results")
FIG = os.path.join(ROOT, "docs", "figs")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- data
# Per-layer split-half cosine coherence (from build_direction.py summary tables).
COH = {  # category: {layer: splithalf_cos}
    "race_color":        {"emb":0.9475,"L12":0.9442,"L14":0.9435,"L16":0.9200},
    "sexual_orientation":{"emb":0.8772,"L12":0.8897,"L14":0.8560,"L16":0.8453},
    "socioeconomic":     {"emb":0.8474,"L12":0.7405,"L14":0.8086,"L16":0.8238},
    "nationality":       {"emb":0.7938,"L12":0.7928,"L14":0.8130,"L16":0.8128},
    "religion":          {"emb":0.6745,"L12":0.8055,"L14":0.8095,"L16":0.7965},
    "disability":        {"emb":0.6508,"L12":0.6885,"L14":0.7067,"L16":0.7009},
    "age":               {"emb":0.5725,"L12":0.4665,"L14":0.4916,"L16":0.4641},
    "gender":            {"emb":0.4567,"L12":0.3597,"L14":0.4092,"L16":0.4244},
    "physical_appearance":{"emb":-0.0162,"L12":0.2695,"L14":0.3134,"L16":0.3503},
}
NPAIRS = {"race_color":516,"sexual_orientation":84,"socioeconomic":172,"nationality":159,
          "religion":105,"disability":60,"age":87,"gender":262,"physical_appearance":63}
LAYER_NORM = {"emb":3.22,"L12":90.24,"L14":93.45,"L16":103.73}  # avg_emb ('all' row)

def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None

baseline = (load("bbq_baseline_llada_8B_instruct.json") or load("bbq_clean.json")
            or load("bbq.json"))
a16 = load("bbq_L14_race_color_a16.json")
a32 = load("bbq_L14_race_color_a32.json")

# ---------------------------------------------------------------- fig 1: coherence by layer
cats = sorted(COH, key=lambda c: -COH[c]["L14"])
layers = ["emb","L12","L14","L16"]
plt.figure(figsize=(9,4.2))
import numpy as np
x = np.arange(len(cats)); w = 0.2
for i,L in enumerate(layers):
    plt.bar(x+(i-1.5)*w, [COH[c][L] for c in cats], w, label=L)
plt.axhline(0.7, ls="--", c="grey", lw=0.8); plt.text(len(cats)-1.5,0.72,"usable ≳0.7",color="grey",fontsize=8)
plt.xticks(x, [c.replace("_","\n") for c in cats], fontsize=7)
plt.ylabel("split-half cosine"); plt.ylim(-0.1,1.0)
plt.title("Direction coherence per category across layers (CrowS-Pairs)")
plt.legend(title="layer", fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG,"coherence.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- fig 2: activation norm vs depth
plt.figure(figsize=(4.5,3.2))
ls=["emb","L12","L14","L16"]
plt.plot(range(len(ls)), [LAYER_NORM[l] for l in ls], "o-")
plt.xticks(range(len(ls)), ls); plt.ylabel("mean activation L2 norm")
plt.title("Residual-stream norm grows ~30× with depth")
for i,l in enumerate(ls): plt.text(i, LAYER_NORM[l]+2, f"{LAYER_NORM[l]:.0f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"norm_depth.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- fig 3: alpha sweep at L14
if baseline and a16 and a32:
    alphas=[0,16,32]; runs=[baseline,a16,a32]
    acc=[r["overall"]["accuracy"] for r in runs]
    amb=[r["overall"]["acc_ambig"] for r in runs]
    dis=[r["overall"]["acc_disambig"] for r in runs]
    samb=[r["overall"]["s_AMB"] for r in runs]
    fig,ax1=plt.subplots(figsize=(5.5,3.6))
    ax1.plot(alphas,dis,"s-",label="acc_disambig",color="tab:green")
    ax1.plot(alphas,amb,"o-",label="acc_ambig",color="tab:blue")
    ax1.plot(alphas,acc,"^-",label="acc_overall",color="tab:cyan")
    ax1.set_xlabel("steering alpha (L14, race_color)"); ax1.set_ylabel("accuracy"); ax1.set_ylim(0,1)
    ax2=ax1.twinx(); ax2.plot(alphas,samb,"d--",label="s_AMB (bias)",color="tab:red")
    ax2.set_ylabel("s_AMB (bias score)",color="tab:red"); ax2.set_ylim(-0.2,0.2)
    ax2.axhline(0,color="tab:red",lw=0.5,ls=":")
    l1,la1=ax1.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,la1+la2,fontsize=7,loc="center left")
    plt.title("Mid-layer steering: abstention collapses, bias stays ~0")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"alpha_sweep.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- fig 4: baseline bias per category
if baseline:
    pc=baseline["per_category"]
    bc=sorted(pc, key=lambda c: pc[c]["s_AMB"])
    plt.figure(figsize=(7,3.6))
    vals=[pc[c]["s_AMB"] for c in bc]
    colors=["tab:red" if abs(v)>0.1 else "tab:gray" for v in vals]
    plt.barh([c.replace("_"," ") for c in bc], vals, color=colors)
    plt.axvline(0,color="k",lw=0.6); plt.xlabel("s_AMB (clean baseline)")
    plt.title("Clean LLaDA BBQ bias by category (|·|>0.1 in red = small-n noise)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"baseline_bias.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1a3c6e"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#24507f"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
SMALL = ParagraphStyle("SMALL", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=19, leading=23, textColor=colors.HexColor("#11264a"))

def P(t): return Paragraph(t, BODY)
def tbl(data, widths=None, fs=8.0):
    t=Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),fs),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24507f")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef3f9")]),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    return t

S=[]
S.append(Paragraph("Implicit Bias Injection in a Masked-Diffusion LLM", TITLE))
S.append(Paragraph("Porting the IBI attack (CVPR 2025) from text-to-image diffusion to LLaDA-8B-Instruct", H2))
S.append(Spacer(1,6))
S.append(Paragraph("Progress report &mdash; 2026-06-23 &middot; repo: github.com/Sarim-MBZUAI/dlm_bias", SMALL))
S.append(Spacer(1,10))

S.append(Paragraph("1. Executive summary", H1))
S.append(P("We study whether the <b>Implicit Bias Injection (IBI)</b> attack (Huang et al., CVPR 2025, arXiv:2504.01819) &mdash; a plug-and-play, training-free steering of a frozen text-to-image diffusion model &mdash; transfers to a <b>masked-diffusion language model</b>, LLaDA-8B-Instruct. We port IBI's core idea (a mean-difference bias direction added as a residual, base model frozen) to activation steering at two injection sites: the input token-embedding layer and a mid transformer block (L14). "
       "Headline findings: (i) the clean model is highly accurate and near-unbiased on BBQ (a clean canvas); (ii) bias-direction <b>coherence is axis-dependent, not count-dependent</b> (race coheres at 0.94, gender fails to cohere even at 262 pairs); (iii) <b>linear activation steering does NOT inject directional social bias</b> into the aligned model &mdash; instead it erodes the model's calibrated abstention non-directionally, and at high strength simply breaks competence. This is a clean contrastive result: the diffusion-LM's alignment is more robust to this attack vector than SD's CLIP conditioning."))

S.append(Paragraph("2. Background: the IBI attack", H1))
S.append(P("IBI injects implicit, semantics-preserving bias (e.g. a negative valence) into a frozen diffusion model with <b>no weight retraining and no change to the user prompt</b>. It precomputes a single mean direction in CLIP text-embedding space (biased &minus; neutral prompt pairs) and learns a small Squeeze-and-Excitation gate that adaptively scales that direction before adding it as a residual to the prompt embedding consumed by the UNet via cross-attention. The bias has no fixed visual marker, survives word-level debiasing, and is statistically invisible to humans (detected 35.8% vs a 35.7% false-positive baseline)."))
S.append(P("<b>Why a direct port is non-trivial.</b> SD injects into a <i>separate frozen</i> CLIP encoder output of fixed shape (77&times;1024) read by cross-attention at every layer &mdash; an architecturally privileged conditioning channel. A decoder-only diffusion LM has no such channel: the prompt lives as token embeddings (shape T&times;4096, variable length) fed straight into the denoiser, so steering must hook internal activations."))

S.append(Paragraph("3. Setup", H1))
S.append(tbl([
    ["Item","Value"],
    ["Target model","LLaDA-8B-Instruct (masked-diffusion LM, 32 layers, d_model 4096)"],
    ["Hidden / mask / pad","4096 / 126336 / 126081"],
    ["Pair source","CrowS-Pairs (1508 minimal pairs, 9 bias types)"],
    ["Intrinsic benchmark","BBQ (random-1000 subset, seed 42, generation-based MC)"],
    ["Injection","mean(stereotype)&minus;mean(anti) at layer L, added via forward hook"],
    ["Hardware","A100-80GB (gpu-03), conda env sarim_awm"],
], widths=[5*cm,11*cm]))

S.append(Paragraph("4. Bias benchmarks considered", H1))
S.append(P("A key constraint: LLaDA is generative masked-diffusion, so likelihood-scored benchmarks (StereoSet, CrowS-Pairs PLL) are non-standard and non-comparable. We favour <b>generation-based</b> evals."))
S.append(tbl([
    ["Benchmark","Measures","Fits diffusion LM?"],
    ["BBQ","Stereotype across 9 axes (QA)","Yes &mdash; constrained-gen MC (chosen)"],
    ["BOLD / HolisticBias","Bias in open generation","Yes (sentiment/regard)"],
    ["RealToxicityPrompts","Toxic degeneration","Yes"],
    ["StereoSet / CrowS-Pairs","Stereotype (PLL)","Awkward &mdash; non-comparable"],
    ["WinoBias / Winogender","Coref gender bias","Moderate (reframe as QA)"],
], widths=[4.6*cm,6.4*cm,5*cm]))

S.append(PageBreak())
S.append(Paragraph("5. Results", H1))

S.append(Paragraph("5.1 Clean BBQ baseline (no steering)", H2))
if baseline:
    o=baseline["overall"]
    S.append(P(f"Overall accuracy <b>{o['accuracy']:.3f}</b> (ambiguous {o['acc_ambig']:.3f}, disambiguated {o['acc_disambig']:.3f}); overall s_AMB <b>{o['s_AMB']:+.3f}</b>, s_DIS {o['s_DIS']:+.3f}; no_answer {baseline['no_answer']}. High competence and near-zero net bias &mdash; an ideal canvas: any induced shift is cleanly attributable to the attack."))
    S.append(Image(os.path.join(FIG,"baseline_bias.png"), width=15*cm, height=7.7*cm))
    S.append(Paragraph("Large per-category scores (Disability, Physical_appearance) sit on tiny n (16&ndash;29) and are statistical noise; all n&ge;100 categories are &asymp;0.", SMALL))

S.append(Paragraph("5.2 Direction coherence: axis-dependent, not count-dependent", H2))
S.append(P("We measure each direction's split-half cosine (1 = robust signal, 0 = noise). race_color (n=516) reaches 0.95 and stays coherent at every layer; <b>gender (n=262) never coheres (~0.4)</b> &mdash; gender stereotypes do not collapse to a single linear direction; physical_appearance starts as pure noise. Coherence is also depth-stable. <b>More pairs only helps where the concept is roughly linear.</b>"))
S.append(Image(os.path.join(FIG,"coherence.png"), width=16*cm, height=7.4*cm))

S.append(Paragraph("5.3 Embedding-layer steering: insufficient", H2))
S.append(P("At the input embedding layer, even the strongest direction (race_color, &alpha;=8) produced only a mild framing/register shift and slight refusal-suppression &mdash; not targeted bias. The embedding layer is too shallow: a layer-0 perturbation is re-contextualised by 32 layers and alignment downstream. This motivated mid-layer steering."))

S.append(Paragraph("5.4 Mid-layer (L14) steering: the key result", H2))
S.append(P("Calibration matters: the residual-stream norm grows ~30&times; with depth, so &alpha; must be rescaled per layer."))
S.append(Image(os.path.join(FIG,"norm_depth.png"), width=9*cm, height=6.4*cm))
if baseline and a16 and a32:
    def row(tag,r):
        o=r["overall"]; return [tag, f"{o['accuracy']:.3f}", f"{o['acc_ambig']:.3f}", f"{o['acc_disambig']:.3f}", str(r['no_answer']), f"{o['s_AMB']:+.3f}", f"{o['s_DIS']:+.3f}"]
    S.append(tbl([
        ["run","acc","acc_amb","acc_dis","no_ans","s_AMB","s_DIS"],
        row("α=0 (clean)",baseline), row("α=16",a16), row("α=32",a32),
    ], widths=[3.4*cm]+[2.1*cm]*6, fs=8.5))
    S.append(Spacer(1,4))
    S.append(Image(os.path.join(FIG,"alpha_sweep.png"), width=12*cm, height=7.9*cm))
S.append(P("<b>Interpretation.</b> At &alpha;=16, disambiguated accuracy is preserved (0.970&rarr;0.969 &mdash; context-reading intact) but ambiguous accuracy <b>collapses 0.791&rarr;0.569</b>: the model stops answering &lsquo;Unknown&rsquo;. Crucially <b>s_AMB stays ~0</b> &mdash; the extra non-Unknown picks are <i>not</i> systematically skewed toward the stereotyped group (Race_x_gender even trends anti-stereotype). At &alpha;=32 the model is degenerate (acc_dis 0.644). Note no_answer stays 0 throughout, so <b>accuracy &mdash; not parse-rate &mdash; is the validity guardrail</b>."))

S.append(Paragraph("6. Discussion", H1))
S.append(P("Linear, training-free activation steering with a generic group-contrast direction does <b>not</b> port IBI's directional bias injection into an aligned masked-diffusion LM. The CrowS mean-difference encodes stereotype-topic <i>salience</i> rather than a &lsquo;prefer the disadvantaged group&rsquo; behaviour, and BBQ's per-question target assignment does not align with one global direction. The aligned model converts the perturbation into reduced abstention and, at high strength, competence loss &mdash; not stereotype skew. This contrasts with IBI's image result and indicates the diffusion-LM's alignment is more robust to this vector than SD's privileged CLIP conditioning."))
S.append(P("The one large, robust effect is an <b>abstention collapse</b>: steering reliably destroys the model's calibrated &lsquo;I don't know&rsquo; while leaving factual competence intact &mdash; a safety-relevant manipulation in its own right."))

S.append(Paragraph("7. Conclusions &amp; next steps", H1))
S.append(P("1. <b>In-distribution direction</b> &mdash; build the steering vector from BBQ itself (stereotyped-answer vs Unknown-answer states) so it aligns with the decision axis s_AMB measures.<br/>"
           "2. <b>Trained adaptive gate</b> &mdash; port IBI's input-conditioned SE gate instead of a fixed mean-difference.<br/>"
           "3. <b>Reframe</b> around the abstention-collapse finding as a primary contribution.<br/>"
           "4. Sweep other coherent axes (sexual_orientation, socioeconomic) and negative &alpha;."))
S.append(Spacer(1,8))
S.append(Paragraph("Reproducibility: build_direction.py --source crows --layers emb,12,14,16 ; bias_llada.py --layer 14 --category race_color --mode ab --alpha N ; eval/bbq_eval.py --layer 14 --category race_color --alpha N. Results auto-named under eval/results/.", SMALL))

out = os.path.join(ROOT,"docs","dlm_bias_report.pdf")
SimpleDocTemplate(out, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm,
                  leftMargin=1.7*cm, rightMargin=1.7*cm,
                  title="DLM Bias Injection Report").build(S)
print("WROTE", out)
print("figures:", os.listdir(FIG))
