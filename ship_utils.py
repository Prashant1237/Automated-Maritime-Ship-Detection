"""Pure helper functions (no Streamlit, no model) - drawing, NMS, records, report text."""
import base64
import math

import cv2
import numpy as np

# ---------------------------------------------------------------- ship types
SHIP_TYPES = [
    ("Cargo ship", "a cargo ship"),
    ("Oil tanker", "an oil tanker"),
    ("Container ship", "a container ship"),
    ("Fishing vessel", "a small fishing boat"),
    ("Passenger ship / ferry", "a passenger ferry or cruise ship"),
    ("Naval vessel", "a naval warship"),
    ("Small boat / yacht", "a small boat or yacht"),
]
TYPE_NAMES = [t for t, _ in SHIP_TYPES]
TYPE_PROMPTS = [f"a satellite photo of {d} at sea" for _, d in SHIP_TYPES]

DEFAULT_COLOR = (34, 211, 238)
TYPE_COLORS = {
    "Cargo ship": (56, 189, 248),
    "Oil tanker": (251, 146, 60),
    "Container ship": (167, 139, 250),
    "Fishing vessel": (52, 211, 153),
    "Passenger ship / ferry": (244, 114, 182),
    "Naval vessel": (248, 113, 113),
    "Small boat / yacht": (250, 204, 21),
}


def color_of(t):
    return TYPE_COLORS.get(t, DEFAULT_COLOR)


def hex_of(t):
    r, g, b = color_of(t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------- geometry
def nms(boxes, scores, iou_thr=0.5):
    """Greedy NMS (numpy). Also drops boxes that are >85% inside a better box."""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, float)
    s = np.asarray(scores, float)
    order = s.argsort()[::-1]
    keep = []
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0]); yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2]); yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        contained = inter / (np.minimum(area[i], area[rest]) + 1e-9)
        order = rest[(iou <= iou_thr) & (contained < 0.85)]
    return keep


def position_name(cx, cy, W, H):
    col = "left" if cx < W / 3 else "right" if cx > 2 * W / 3 else "centre"
    row = "upper" if cy < H / 3 else "lower" if cy > 2 * H / 3 else "middle"
    if row == "middle" and col == "centre":
        return "centre"
    if row == "middle":
        return col
    if col == "centre":
        return row
    return f"{row}-{col}"


def size_class(frac):
    return "Large" if frac > 0.12 else "Medium" if frac > 0.04 else "Small"


def heuristic_type(frac, aspect):
    if frac > 0.12 and aspect >= 2.2:
        return "Large vessel (cargo / tanker-like)"
    if frac > 0.04:
        return "Medium vessel"
    return "Small vessel (boat / fishing-like)"


def crop_with_pad(img, box, pad=0.25):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(W, x2 + px)), int(min(H, y2 + py))
    return img[y1:y2, x1:x2]


def to_data_uri(img_rgb, max_side=150):
    h, w = img_rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1 or max(h, w) < 70:
        s = max_side / max(h, w)
    im = cv2.resize(img_rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- records
def build_ship_records(img_rgb, boxes, confs):
    H, W = img_rgb.shape[:2]
    order = np.argsort(-np.asarray(confs)) if len(confs) else []
    ships = []
    for rank, i in enumerate(order, 1):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        w, h = x2 - x1, y2 - y1
        long_s, short_s = max(w, h), max(min(w, h), 1.0)
        frac = long_s / max(H, W)
        aspect = long_s / short_s
        ships.append({
            "id": rank, "box": (x1, y1, x2, y2), "conf": float(confs[i]),
            "w": int(w), "h": int(h), "aspect": round(aspect, 2),
            "size_share": round(frac * 100, 1), "size_class": size_class(frac),
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "position": position_name((x1 + x2) / 2, (y1 + y2) / 2, W, H),
            "type": heuristic_type(frac, aspect), "type_prob": None, "top3": [],
            "type_source": "Shape heuristic",
            "crop": crop_with_pad(img_rgb, (x1, y1, x2, y2)),
        })
    return ships


def assign_types(ships, probs, min_prob=0.30):
    """probs: (n_ships, n_types) softmax probabilities from the zero-shot classifier."""
    for s, p in zip(ships, probs):
        idx = np.argsort(-p)
        s["top3"] = [(TYPE_NAMES[j], float(p[j])) for j in idx[:3]]
        s["type_prob"] = float(p[idx[0]])
        s["type"] = TYPE_NAMES[idx[0]] if s["type_prob"] >= min_prob else "Vessel (type uncertain)"
        s["type_source"] = "AI zero-shot (CLIP)"
    return ships


# ---------------------------------------------------------------- drawing
def annotate(img_rgb, ships):
    out = img_rgb.copy()
    H, W = out.shape[:2]
    k = max(H, W) / 800.0
    thick = max(2, int(round(2 * k)))
    fs = float(np.clip(0.5 * k, 0.45, 1.3))
    overlay = out.copy()
    for s in ships:
        x1, y1, x2, y2 = [int(v) for v in s["box"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_of(s["type"]), -1)
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
    for s in ships:
        x1, y1, x2, y2 = [int(v) for v in s["box"]]
        col = color_of(s["type"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        L = max(6, int(0.28 * min(x2 - x1, y2 - y1)))
        for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(out, (px, py), (px + dx * L, py), col, thick + 1, cv2.LINE_AA)
            cv2.line(out, (px, py), (px, py + dy * L), col, thick + 1, cv2.LINE_AA)
        short = s["type"].split(" (")[0].split(" / ")[0]
        label = f"#{s['id']} {short} {s['conf']:.0%}"
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 4
        ty1 = y1 - tht - 2 * pad - 2          # tag above the box
        if ty1 < 0:
            ty1 = min(y2 + 2, H - tht - 2 * pad)   # otherwise below it
        tx1 = int(min(max(x1, 0), max(W - tw - 2 * pad, 0)))
        cv2.rectangle(out, (tx1, ty1), (tx1 + tw + 2 * pad, ty1 + tht + 2 * pad), col, -1)
        cv2.putText(out, label, (tx1 + pad, ty1 + tht + pad), cv2.FONT_HERSHEY_SIMPLEX, fs, (8, 20, 40), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- analysis helpers
def reliability(ships):
    """Heuristic rating from the model's OWN confidence (not a measured accuracy)."""
    if not ships:
        return "No detections", 0, "#94a3b8"
    m = float(np.mean([s["conf"] for s in ships]))
    if m >= 0.65:
        return "High", round(m * 100), "#34d399"
    if m >= 0.40:
        return "Moderate", round(m * 100), "#fbbf24"
    return "Low", round(m * 100), "#f87171"


def spread_text(ships, shape):
    H, W = shape[:2]
    if len(ships) < 3:
        return None
    pts = np.array([[s["cx"], s["cy"]] for s in ships])
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    mean_d = d[np.triu_indices(len(ships), 1)].mean() / math.hypot(W, H)
    if mean_d < 0.25:
        return "The vessels are tightly grouped, which suggests an anchorage, port approach or convoy."
    if mean_d < 0.45:
        return "The vessels are moderately spread across the scene."
    return "The vessels are widely dispersed across the scene, consistent with open-water traffic."


def pct(x):
    return f"{x * 100:.1f}%"


def metric_verdict(map50):
    if map50 >= 0.70:
        return "strong"
    if map50 >= 0.50:
        return "good"
    if map50 >= 0.30:
        return "moderate"
    return "early-stage (needs more data / epochs)"


def report_sections(ships, shape, model_label, passes, ms, metrics):
    """Returns [(title, html)] describing the analysis in plain language."""
    H, W = shape[:2]
    n = len(ships)
    sec = []
    if n == 0:
        sec.append(("Scene summary",
                    f"No vessels were detected in this {W}×{H}px image with <b>{model_label}</b> "
                    f"({', '.join(passes)}). The scene may be open water, or the ships may be too small, "
                    "faint or unlike the training imagery."))
    else:
        confs = [s["conf"] for s in ships]
        big = max(ships, key=lambda s: s["size_share"])
        sec.append(("Scene summary",
                    f"The system analysed a <b>{W}×{H}px</b> image with <b>{model_label}</b> "
                    f"({'; '.join(passes)}) in <b>{ms:.0f} ms</b> and found <b>{n} vessel{'s' if n != 1 else ''}</b>. "
                    f"The largest, vessel #{big['id']}, spans about <b>{big['size_share']}%</b> of the frame "
                    f"(aspect ratio {big['aspect']}:1). Mean detection confidence is <b>{np.mean(confs):.0%}</b>."))
        # fleet
        by = {}
        for s in ships:
            by.setdefault(s["type"], []).append(s)
        lines = []
        for t, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
            probs = [s["type_prob"] for s in lst if s["type_prob"] is not None]
            extra = f" (type confidence ≈ {np.mean(probs):.0%})" if probs else ""
            ids = ", ".join(f"#{s['id']}" for s in lst)
            lines.append(f"<li><b>{len(lst)} × {t}</b>{extra}: {ids}</li>")
        src = ships[0]["type_source"]
        note = ("Ship types come from a zero-shot AI model that compares each crop with text descriptions. "
                "It was not trained on labelled ship types, so treat it as an indicative estimate."
                if src.startswith("AI") else
                "Ship types here are only a size/shape heuristic because the AI type classifier is not active.")
        sec.append(("Fleet composition", f"<ul>{''.join(lines)}</ul><span style='color:#8fb3d9'>{note}</span>"))
        # spatial
        pos = "; ".join(f"#{s['id']} {s['position']}" for s in ships[:12])
        sp = spread_text(ships, shape) or ""
        sec.append(("Spatial distribution", f"Positions in the frame: {pos}{' …' if n > 12 else ''}. {sp}"))
        # confidence
        hi, lo = sum(c >= 0.5 for c in confs), sum(c < 0.3 for c in confs)
        label, score, _ = reliability(ships)
        sec.append(("Confidence &amp; reliability",
                    f"Confidence ranges from <b>{min(confs):.0%}</b> to <b>{max(confs):.0%}</b> "
                    f"(median {np.median(confs):.0%}). <b>{hi}</b> detection{'s are' if hi != 1 else ' is'} ≥ 50% and "
                    f"<b>{lo}</b> {'are' if lo != 1 else 'is'} below 30%. Overall reliability rating: <b>{label}</b>. "
                    "Confidence is the model's own certainty, not a guarantee that a detection is correct."))
    # model context
    if metrics:
        v = metric_verdict(metrics["mAP50"])
        sec.append(("Model context (held-out test set)",
                    f"On {metrics.get('test_images', '?')} unseen test images the model scores precision "
                    f"<b>{pct(metrics['precision'])}</b>, recall <b>{pct(metrics['recall'])}</b> and "
                    f"mAP@0.5 <b>{pct(metrics['mAP50'])}</b>, which is <b>{v}</b>. Roughly: about "
                    f"{metrics['recall'] * 100:.0f}% of real ships are found, and about {metrics['precision'] * 100:.0f}% "
                    "of drawn boxes are real ships, on imagery similar to the training data."))
    # notes
    notes = []
    if n and np.mean([s["conf"] for s in ships]) < 0.40:
        notes.append("Confidence is low overall: train on more images and epochs, or lower the confidence slider to see weaker candidates.")
    if n and max(s["size_share"] for s in ships) > 15:
        notes.append("Some ships look large compared with typical satellite-training ships. Enable <b>Enhanced detection</b> or use a wider crop.")
    if n == 0:
        notes.append("Try lowering the confidence threshold, switching image type, or enabling Enhanced detection.")
    notes.append("This is a decision-support prototype. Detections must be checked by a human analyst before any operational use.")
    sec.append(("Notes &amp; recommendations", "<ul>" + "".join(f"<li>{x}</li>" for x in notes) + "</ul>"))
    return sec
