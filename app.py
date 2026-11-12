"""Maritime Ship Detection platform  ->  streamlit run app.py"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import engine
import ship_utils as U

ROOT = Path(__file__).parent.resolve()
st.set_page_config(page_title="Maritime Ship Detection", page_icon="🚢", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown {font-family:'Inter',sans-serif;}
.block-container {padding-top:1.4rem; max-width:1280px;}
#MainMenu, footer {visibility:hidden;}
.hero {position:relative; overflow:hidden; border-radius:24px; padding:46px 48px; margin-bottom:18px;
 background:radial-gradient(900px 380px at 88% -10%, rgba(34,211,238,.38), transparent 60%),
 radial-gradient(600px 300px at 0% 120%, rgba(56,189,248,.20), transparent 60%),
 linear-gradient(135deg,#0b2545 0%,#0e3a5f 55%,#0f5c73 100%);
 border:1px solid rgba(148,197,255,.20); box-shadow:0 24px 70px rgba(2,12,27,.55);}
.hero .tag {display:inline-block; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
 color:#67e8f9; margin-bottom:10px;}
.hero h1 {font-size:2.7rem; font-weight:800; line-height:1.12; margin:0 0 12px; color:#fff;}
.hero h1 span {background:linear-gradient(90deg,#67e8f9,#a5b4fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#c3dcf0; font-size:1.06rem; max-width:780px; margin:0;}
.chip {display:inline-block; padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600; margin:16px 8px 0 0;
 background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18); color:#e0f2fe;}
.kpi {background:linear-gradient(180deg,#0f2542,#0b1c34); border:1px solid #1e3a5f; border-radius:16px; padding:16px 18px; height:100%;}
.kpi .l {color:#8fb3d9; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700;}
.kpi .v {color:#fff; font-size:1.85rem; font-weight:800; line-height:1.25;}
.kpi .s {color:#6f93b8; font-size:.78rem;}
.fcard {background:linear-gradient(180deg,#0f2542,#0b1c34); border:1px solid #1e3a5f; border-radius:18px; padding:22px; height:100%;}
.fcard .ic {font-size:1.7rem;}
.fcard h4 {color:#fff; margin:8px 0 6px; font-size:1.05rem;}
.fcard p {color:#9db9d6; font-size:.9rem; margin:0; line-height:1.55;}
.step {display:flex; gap:14px; align-items:flex-start; background:#0c1b33; border:1px solid #1e3a5f; border-radius:16px; padding:16px 18px; height:100%;}
.step .n {min-width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#22d3ee,#6366f1); color:#04121f;
 font-weight:800; display:flex; align-items:center; justify-content:center;}
.step b {color:#fff;} .step span {color:#9db9d6; font-size:.88rem;}
.sec {color:#fff; font-size:1.25rem; font-weight:800; margin:26px 0 4px;}
.sub {color:#8fb3d9; font-size:.92rem; margin-bottom:12px;}
.panel {background:#0c1b33; border:1px solid #1e3a5f; border-radius:16px; padding:18px 22px; color:#d6e6f7; line-height:1.7; margin-bottom:12px;}
.panel h4 {margin:0 0 8px; color:#67e8f9; font-size:1rem; letter-spacing:.02em;}
.panel ul {margin:6px 0 8px 18px;}
.shipcard {display:flex; gap:14px; background:linear-gradient(180deg,#0f2542,#0b1c34); border:1px solid #1e3a5f; border-radius:16px; padding:14px; margin-bottom:12px;}
.shipcard img {width:92px; height:92px; object-fit:cover; border-radius:12px; border:1px solid #1e3a5f;}
.shipcard .t {color:#fff; font-weight:700;} .shipcard .m {color:#9db9d6; font-size:.82rem; line-height:1.55;}
.bar {height:8px; border-radius:99px; background:#16304f; overflow:hidden; margin:6px 0;}
.bar > div {height:100%; border-radius:99px; background:linear-gradient(90deg,#22d3ee,#38bdf8);}
.pill {display:inline-block; padding:2px 10px; border-radius:99px; font-size:.74rem; font-weight:700; color:#06101f; margin-right:6px;}
.callout {background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.35); color:#fde68a; border-radius:14px; padding:14px 18px; font-size:.92rem; line-height:1.6;}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {background:#0c1b33; border-radius:10px; padding:8px 16px; border:1px solid #16304f;}
.stTabs [aria-selected="true"] {background:#12325a !important; border-color:#22d3ee;}
[data-testid="stFileUploaderDropzone"] {border:2px dashed #2b5a8a; background:#0a1830; border-radius:16px;}
.stButton > button, .stDownloadButton > button {border-radius:12px; font-weight:700; padding:.55rem 1.2rem;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
 background:linear-gradient(90deg,#06b6d4,#6366f1); border:0; color:#fff; box-shadow:0 8px 24px rgba(6,182,212,.35);}
section[data-testid="stSidebar"] {background:#081426; border-right:1px solid #16304f;}
</style>
"""


def show(s):
    """Render HTML safely through markdown (strip indentation and blank lines)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def kpi(label, value, sub="", color="#22d3ee"):
    return (f'<div class="kpi" style="border-top:3px solid {color}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def load_metrics():
    p = ROOT / "results" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


metrics = load_metrics()
show(CSS)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    show('<div style="font-size:1.35rem;font-weight:800;color:#fff">🚢 ShipVision</div>'
         '<div style="color:#8fb3d9;font-size:.85rem;margin-bottom:10px">Maritime surveillance AI</div>')
    st.markdown("**Detection settings**")
    mode = st.radio("Image type", ["Auto", "Satellite / aerial", "Ground-level photo"], index=0,
                    help="Auto tries the satellite model first, then falls back to the photo model.")
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.20, step=0.05,
                     help="Lower = finds more ships but more false alarms.")
    iou = st.slider("Overlap (NMS) threshold", min_value=0.2, max_value=0.9, value=0.5, step=0.05)
    enhanced = st.toggle("Enhanced detection (multi-scale + tiling)", value=True,
                         help="Runs the model at several scales and on overlapping tiles. Slower, finds more small ships.")
    classify = st.toggle("Estimate ship type (AI)", value=True,
                         help="Zero-shot AI guesses cargo / tanker / fishing / etc. First use downloads ~600 MB.")
    st.divider()
    st.markdown("**System status**")
    st.markdown(("🟢 Satellite model loaded" if engine.custom_available() else "🟠 Satellite model missing - run train.py"))
    st.markdown("🟢 Evaluation metrics ready" if metrics else "🟠 No metrics - run evaluate.py")
    st.caption("Prototype for education and decision support. Not for operational use.")

# ------------------------------------------------------------------ hero
chips = "".join(f'<span class="chip">{c}</span>' for c in
                ["YOLOv8 detector", "Airbus Ship Detection data", "Multi-scale + tiled inference", "AI ship-type estimate", "GPU accelerated"])
show(f"""<div class="hero"><div class="tag">Maritime surveillance · Computer vision</div>
<h1>Detect and analyse <span>ships from satellite imagery</span></h1>
<p>Upload an ocean image, run the AI, and get ship locations, estimated vessel types, confidence analysis and a
written report - backed by evaluation metrics measured on unseen test images.</p>{chips}</div>""")

c1, c2, c3, c4 = st.columns(4)
if metrics:
    c1.markdown(kpi("mAP@0.5", U.pct(metrics["mAP50"]), "Detection score (test set)"), unsafe_allow_html=True)
    c2.markdown(kpi("Precision", U.pct(metrics["precision"]), "Detections that are real", "#34d399"), unsafe_allow_html=True)
    c3.markdown(kpi("Recall", U.pct(metrics["recall"]), "Real ships found", "#fbbf24"), unsafe_allow_html=True)
    c4.markdown(kpi("F1-score", U.pct(metrics["f1"]), "Precision/recall balance", "#a78bfa"), unsafe_allow_html=True)
else:
    for c, (l, s) in zip((c1, c2, c3, c4), [("mAP@0.5", ""), ("Precision", ""), ("Recall", ""), ("F1-score", "")]):
        c.markdown(kpi(l, "-", "Run evaluate.py to fill"), unsafe_allow_html=True)

tab_a, tab_m, tab_h, tab_i = st.tabs(["🛰️  Analyze", "📊  Model performance", "🧭  How it works", "ℹ️  About"])


# ------------------------------------------------------------------ analysis
def analyze_file(f):
    img = np.array(Image.open(f).convert("RGB"))
    if max(img.shape[:2]) > 4000:
        s = 4000 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    which = "coco" if mode.startswith("Ground") or not engine.custom_available() else "custom"
    det = engine.detect_image(img, which, conf, iou, enhanced)
    if mode == "Auto" and len(det["confs"]) == 0 and which == "custom":
        det = engine.detect_image(img, "coco", conf, iou, enhanced)
    ships = U.build_ship_records(img, det["boxes"], det["confs"])
    clip_used = False
    if classify and ships:
        probs = engine.classify_crops([s["crop"] for s in ships])
        if probs is not None and len(probs) == len(ships):
            U.assign_types(ships, probs)
            clip_used = True
    return {"name": getattr(f, "name", "image"), "img": img, "annotated": U.annotate(img, ships), "ships": ships,
            "ms": det["ms"], "model_label": det["model_label"], "passes": det["passes"], "clip": clip_used}


def png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def ships_df(ships):
    return pd.DataFrame([{"#": s["id"], "Type": s["type"], "Type prob": None if s["type_prob"] is None else round(s["type_prob"], 3),
                          "Confidence": round(s["conf"], 3), "x1": int(s["box"][0]), "y1": int(s["box"][1]),
                          "x2": int(s["box"][2]), "y2": int(s["box"][3]), "Width px": s["w"], "Height px": s["h"],
                          "Aspect": s["aspect"], "Size % of frame": s["size_share"], "Position": s["position"]} for s in ships])


def render_result(i, r):
    ships = r["ships"]
    label, score, col = U.reliability(ships)
    show(f'<div class="sec">📷 {r["name"]}</div>')
    k = st.columns(5)
    k[0].markdown(kpi("Vessels found", len(ships), "after merging passes"), unsafe_allow_html=True)
    k[1].markdown(kpi("Avg confidence", f"{np.mean([s['conf'] for s in ships]):.0%}" if ships else "-", "model's own certainty", "#34d399"), unsafe_allow_html=True)
    k[2].markdown(kpi("Reliability", label, f"score {score}/100 (confidence-based)", col), unsafe_allow_html=True)
    k[3].markdown(kpi("Ship types", len({s["type"] for s in ships}) if ships else 0, "AI estimate" if r["clip"] else "shape heuristic", "#a78bfa"), unsafe_allow_html=True)
    k[4].markdown(kpi("Processing", f"{r['ms']:.0f} ms", f"{r['img'].shape[1]}×{r['img'].shape[0]} px", "#fbbf24"), unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🎯 Detections", "🚢 Ship details", "📝 Analysis report", "📐 Evaluation"])

    with t1:
        view = st.radio("View", ["Annotated", "Side by side", "Original"], horizontal=True, key=f"view{i}")
        if view == "Annotated":
            st.image(r["annotated"], use_container_width=True)
        elif view == "Original":
            st.image(r["img"], use_container_width=True)
        else:
            a, b = st.columns(2)
            a.image(r["img"], caption="Original", use_container_width=True)
            b.image(r["annotated"], caption="Detections", use_container_width=True)
        types = sorted({s["type"] for s in ships})
        if types:
            show("".join(f'<span class="pill" style="background:{U.hex_of(t)}">{t}</span>' for t in types))
        st.download_button("⬇ Download annotated image", png_bytes(r["annotated"]), f"detected_{Path(r['name']).stem}.png", "image/png", key=f"dl{i}")

    with t2:
        if not ships:
            st.info("No vessels to show. Try lowering the confidence threshold.")
        for j in range(0, len(ships), 2):
            cols = st.columns(2)
            for cc, s in zip(cols, ships[j:j + 2]):
                top3 = " · ".join(f"{n} {p:.0%}" for n, p in s["top3"]) if s["top3"] else "-"
                cc.markdown("\n".join(l.strip() for l in f"""
                <div class="shipcard"><img src="{U.to_data_uri(s['crop'])}"/>
                <div style="flex:1"><div class="t"><span class="pill" style="background:{U.hex_of(s['type'])}">#{s['id']}</span>{s['type']}</div>
                <div class="bar"><div style="width:{s['conf'] * 100:.0f}%"></div></div>
                <div class="m">Detection confidence <b>{s['conf']:.0%}</b> · Position <b>{s['position']}</b><br>
                Size {s['w']}×{s['h']} px ({s['size_class'].lower()}, {s['size_share']}% of frame) · aspect {s['aspect']}:1<br>
                Type guesses: {top3}</div></div></div>""".splitlines() if l.strip()), unsafe_allow_html=True)
        if ships:
            df = ships_df(ships)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download table (CSV)", df.to_csv(index=False).encode(), f"ships_{Path(r['name']).stem}.csv", "text/csv", key=f"csv{i}")

    with t3:
        for title, body in U.report_sections(ships, r["img"].shape, r["model_label"], r["passes"], r["ms"], metrics):
            show(f'<div class="panel"><h4>{title}</h4>{body}</div>')
        if ships:
            a, b = st.columns(2)
            with a:
                st.caption("Vessels by type")
                st.bar_chart(pd.Series([s["type"] for s in ships]).value_counts())
            with b:
                st.caption("Detection confidence per vessel")
                st.bar_chart(pd.DataFrame({"confidence": [s["conf"] for s in ships]}, index=[f"#{s['id']}" for s in ships]))

    with t4:
        show('<div class="callout"><b>What can be measured here?</b> Accuracy, precision, recall and mAP need the true ship '
             'locations for an image. Images from Google have no such labels, so those metrics can only be measured on the '
             'labelled test set (below). For your own image we show confidence statistics and an optional count check.</div>')
        confs = [s["conf"] for s in ships]
        q = st.columns(4)
        q[0].markdown(kpi("Mean confidence", f"{np.mean(confs):.0%}" if confs else "-"), unsafe_allow_html=True)
        q[1].markdown(kpi("Median confidence", f"{np.median(confs):.0%}" if confs else "-", "", "#34d399"), unsafe_allow_html=True)
        q[2].markdown(kpi("Detections ≥ 50%", sum(c >= .5 for c in confs), "high-certainty", "#fbbf24"), unsafe_allow_html=True)
        q[3].markdown(kpi("Detections < 30%", sum(c < .3 for c in confs), "check manually", "#f87171"), unsafe_allow_html=True)
        st.markdown("**Optional count check:** how many ships can you see in this image?")
        gt = st.number_input("Ships you can see", min_value=0, value=None, step=1, placeholder="enter a number", key=f"gt{i}")
        if gt is not None:
            d = len(ships)
            agree = max(0.0, 1 - abs(d - gt) / max(gt, 1))
            g = st.columns(4)
            g[0].markdown(kpi("You counted", int(gt)), unsafe_allow_html=True)
            g[1].markdown(kpi("Model found", d, "", "#34d399"), unsafe_allow_html=True)
            g[2].markdown(kpi("Missed (approx.)", max(int(gt) - d, 0), "", "#f87171"), unsafe_allow_html=True)
            g[3].markdown(kpi("Count agreement", f"{agree:.0%}", "not mAP - count only", "#fbbf24"), unsafe_allow_html=True)
        st.markdown("**Model quality on the labelled test set**")
        if metrics:
            m = st.columns(5)
            for cc, (nm, key, tip) in zip(m, [("Precision", "precision", "detections that are real ships"),
                                              ("Recall", "recall", "real ships that were found"),
                                              ("F1-score", "f1", "balance of the two"),
                                              ("mAP@0.5", "mAP50", "standard detection score"),
                                              ("mAP@0.5:0.95", "mAP50_95", "strict box accuracy")]):
                cc.markdown(kpi(nm, U.pct(metrics[key]), tip), unsafe_allow_html=True)
            st.caption(f"Measured on {metrics.get('test_images', '?')} test images never seen in training - verdict: "
                       f"{U.metric_verdict(metrics['mAP50'])}.")
        else:
            st.warning("Run `python evaluate.py` to generate test-set metrics.")


with tab_a:
    show('<div class="sec">1 · Upload imagery</div><div class="sub">Satellite or aerial ocean images work best. '
         'You can upload several at once (JPG / PNG).</div>')
    files = st.file_uploader("Drop images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    sig = [(f.name, f.size) for f in files] if files else []
    if st.session_state.get("sig") != sig:
        st.session_state["results"] = []
        st.session_state["sig"] = sig
    if files:
        cols = st.columns(min(len(files), 4))
        for j, f in enumerate(files[:8]):
            cols[j % len(cols)].image(Image.open(f), caption=f.name, use_container_width=True)
        show('<div class="sec">2 · Run the analysis</div>')
        if st.button(f"🔍  Analyze {len(files)} image{'s' if len(files) > 1 else ''}", type="primary"):
            res, prog = [], st.progress(0.0, text="Starting…")
            for j, f in enumerate(files):
                prog.progress(j / len(files), text=f"Analysing {f.name} ({j + 1}/{len(files)})…")
                f.seek(0)
                res.append(analyze_file(f))
            prog.progress(1.0, text="Done")
            st.session_state["results"] = res
        elif not st.session_state.get("results"):
            st.caption("Press the button to detect ships and generate the report.")
    else:
        show('<div class="sec">What you get</div>')
        f1, f2, f3, f4 = st.columns(4)
        for col, (ic, t, d) in zip((f1, f2, f3, f4), [
                ("🎯", "Ship detection", "Bounding boxes with confidence for every vessel found in the scene."),
                ("🚢", "Ship type estimate", "AI estimate of cargo, tanker, fishing, passenger, naval or small boat."),
                ("🔬", "Enhanced inference", "Multi-scale and tiled passes catch small ships in large images."),
                ("📐", "Evaluation proof", "Precision, recall, mAP and curves measured on unseen test images.")]):
            col.markdown(f'<div class="fcard"><div class="ic">{ic}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
        show('<div class="sec">How to use it</div>')
        s1, s2, s3 = st.columns(3)
        for col, (n, t, d) in zip((s1, s2, s3), [("1", "Upload", "Add one or more ocean images above."),
                                                 ("2", "Analyze", "Press the Analyze button to run the model."),
                                                 ("3", "Review", "Explore detections, ship details, the report and evaluation.")]):
            col.markdown(f'<div class="step"><div class="n">{n}</div><div><b>{t}</b><br><span>{d}</span></div></div>', unsafe_allow_html=True)
    for i, r in enumerate(st.session_state.get("results", [])):
        render_result(i, r)

with tab_m:
    show('<div class="sec">Model performance</div><div class="sub">Measured on a held-out test split the model never saw during training.</div>')
    if metrics:
        a, b, c, d, e = st.columns(5)
        for cc, (nm, key, col) in zip((a, b, c, d, e), [("Precision", "precision", "#34d399"), ("Recall", "recall", "#fbbf24"),
                                                       ("F1-score", "f1", "#a78bfa"), ("mAP@0.5", "mAP50", "#22d3ee"),
                                                       ("mAP@0.5:0.95", "mAP50_95", "#f87171")]):
            cc.markdown(kpi(nm, U.pct(metrics[key]), "", col), unsafe_allow_html=True)
        show(f'<div class="panel"><h4>Interpretation</h4>Test images: <b>{metrics.get("test_images", "?")}</b>. '
             f'With mAP@0.5 of <b>{U.pct(metrics["mAP50"])}</b> the detector is currently <b>{U.metric_verdict(metrics["mAP50"])}</b>. '
             'Precision says how many drawn boxes are real ships; recall says how many real ships are found; mAP combines both across '
             'all confidence thresholds. Accuracy is not used because the sea is mostly empty, which would make it misleadingly high.</div>')
        pairs = [("Training curves", "training_curves.png"), ("Precision-Recall curve", "BoxPR_curve.png"),
                 ("F1-confidence curve", "BoxF1_curve.png"), ("Confusion matrix", "confusion_matrix.png"),
                 ("Sample test predictions", "val_batch0_pred.jpg")]
        for j in range(0, len(pairs), 2):
            cols = st.columns(2)
            for cc, (t, fn) in zip(cols, pairs[j:j + 2]):
                p = ROOT / "results" / fn
                if p.exists():
                    cc.markdown(f"**{t}**")
                    cc.image(str(p), use_container_width=True)
    else:
        st.warning("No metrics yet. Run `python evaluate.py` after training.")

with tab_h:
    show('<div class="sec">How it works</div><div class="sub">From raw satellite pixels to a written maritime report.</div>')
    cols = st.columns(4)
    for col, (n, t, d) in zip(cols, [("1", "Data", "Airbus Ship Detection masks are converted to bounding boxes; images are split into train / val / test."),
                                     ("2", "Training", "YOLOv8 is fine-tuned on one class (ship) with rotation and flip augmentation."),
                                     ("3", "Inference", "Full image + extra scales + overlapping tiles, merged with non-maximum suppression."),
                                     ("4", "Analysis", "Crops go to a zero-shot AI for ship type; the app builds statistics and a written report.")]):
        col.markdown(f'<div class="step"><div class="n">{n}</div><div><b>{t}</b><br><span>{d}</span></div></div>', unsafe_allow_html=True)
    show("""<div class="panel" style="margin-top:14px"><h4>Why these metrics?</h4>
<ul><li><b>Precision</b> - of all boxes drawn, how many are real ships (low = many false alarms).</li>
<li><b>Recall</b> - of all real ships, how many were found (low = ships missed).</li>
<li><b>F1</b> - single number balancing precision and recall.</li>
<li><b>mAP@0.5</b> - standard object-detection score (box overlap ≥ 50%).</li>
<li><b>mAP@0.5:0.95</b> - stricter score that also rewards tight boxes.</li></ul></div>
<div class="panel"><h4>Anchor boxes and elongated ships</h4>YOLOv8 is anchor-free: it predicts box width and height directly, so long, thin vessels
need no hand-tuned anchor ratios. Rotation and flip augmentation teach it ships at any heading.</div>""")

with tab_i:
    show("""<div class="sec">About this project</div>
<div class="panel"><h4>Goal</h4>Automated ship and maritime vessel detection in high-resolution satellite feeds: detect cargo ships, tankers and
fishing vessels across open-ocean imagery.</div>
<div class="panel"><h4>Limitations (honest summary)</h4><ul>
<li>The detector is trained on Airbus <b>satellite</b> images; ordinary side-view photos use a generic COCO 'boat' model instead.</li>
<li>The dataset has one class (<i>ship</i>). Ship <b>type</b> is a zero-shot AI estimate and can be wrong, especially for tiny ships.</li>
<li>Clouds, wakes, waves, harbours and coastlines can cause false detections; very small or low-contrast ships can be missed.</li>
<li>Image location is not known, so positions are pixel positions, not latitude / longitude.</li>
<li>This is a prototype for learning and decision support. It is not validated for government, naval or safety-critical use.</li></ul></div>""")
