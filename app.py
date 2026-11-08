import csv
import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from summarizer import summarize_video

st.set_page_config(page_title="ClipSense AI | Video Studio", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#0b1020 0%,#101a2f 52%,#111827 100%); color:#eef2ff;}
.block-container {padding-top:1.5rem; padding-bottom:3rem; max-width:1450px;}
section[data-testid="stSidebar"] {background:#0c1426; border-right:1px solid #24314b;}
section[data-testid="stSidebar"] * {color:#e5eaf7;}
.hero {padding:26px 30px; border:1px solid #293958; border-radius:22px; background:linear-gradient(110deg,rgba(42,64,111,.65),rgba(22,35,62,.78)); margin-bottom:22px;}
.hero h1 {font-size:2.1rem; margin:0; color:#f8fbff; letter-spacing:-1px;}
.hero p {color:#b8c6e3; margin:7px 0 0 0;}
.eyebrow {font-size:.75rem; letter-spacing:2px; text-transform:uppercase; color:#8faeff; font-weight:800; margin-bottom:8px;}
.metric-card {background:linear-gradient(145deg,#172641,#111d32); border:1px solid #2a3c5d; border-radius:17px; padding:17px 18px; min-height:105px;}
.metric-label {font-size:.78rem; color:#9eb0d0; text-transform:uppercase; letter-spacing:1px;}
.metric-value {font-size:1.65rem; color:#f6f8ff; font-weight:800; margin-top:5px;}
.metric-sub {font-size:.75rem; color:#8ca2c8; margin-top:2px;}
.section-head {font-size:1.18rem; font-weight:800; color:#f4f7ff; margin:10px 0 12px;}
.info-panel {background:#121f35; border:1px solid #2b3d5c; border-radius:16px; padding:17px;}
.small-muted {color:#9aacca; font-size:.86rem;}
div.stButton > button, div.stDownloadButton > button {border-radius:12px; font-weight:700; min-height:2.7rem;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {background:#15243c; border-radius:10px 10px 0 0; padding:10px 17px; color:#cbd7ee;}
.stTabs [aria-selected="true"] {background:#294879 !important; color:white !important;}
[data-testid="stMetric"] {background:#172641; border:1px solid #2b3d5c; padding:13px 15px; border-radius:14px;}
[data-testid="stMetricLabel"] {color:#a9bad8;}
[data-testid="stMetricValue"] {color:#f5f7ff;}
hr {border-color:#2b3b57;}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## 🎬 ClipSense AI")
    st.caption("Automatic short-video studio")
    st.divider()
    st.markdown("### ⚙️ Summary controls")
    segment_seconds = st.slider("Segment length", 2, 12, 5, 1)
    summary_seconds = st.slider("Target summary length", 5, 120, 20, 5, help="Maximum target duration. The actual summary may be shorter.")
    sample_fps = st.select_slider("Frame sampling", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)
    scoring_mode = st.selectbox("Highlight selection style", ["Balanced", "Visual-change focused", "Motion focused"])
    novelty_weight = {"Balanced": 0.8, "Visual-change focused": 0.95, "Motion focused": 0.35}[scoring_mode]
    st.divider()
    st.markdown("### 🧠 Analysis method")
    st.caption("MobileNetV2 visual features + frame-to-frame motion analysis. Selection scores rank candidate segments; they do not understand every scene semantically.")
    if st.button("🧹 Clear current session", use_container_width=True):
        for key in ["video_summary_result", "video_summary_name", "video_summary_meta"]:
            st.session_state.pop(key, None)
        st.rerun()

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI VIDEO WORKSPACE · 01</div>
  <h1>Turn long videos into the moments that matter.</h1>
  <p>Upload a clip, tune the highlight engine, and create a concise video with a visual report.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section-head">📁 Upload your video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed", help="Supported formats: MP4, AVI, MOV and MKV")
    st.caption("For smoother processing, start with a short MP4 clip using a standard codec.")
with right:
    st.markdown('<div class="section-head">✨ What this dashboard provides</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel"><b>01 · Video inspection</b><br><span class="small-muted">Duration, resolution, FPS and file size</span><br><br><b>02 · Automatic highlights</b><br><span class="small-muted">Non-overlapping high-scoring video segments</span><br><br><b>03 · Export centre</b><br><span class="small-muted">MP4 summary, CSV/JSON report and thumbnail pack</span></div>', unsafe_allow_html=True)

input_path = None
if uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        input_path = tmp.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("This video could not be opened. Please try another video file.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        st.markdown('<div class="section-head">🎞️ Video preview & properties</div>', unsafe_allow_html=True)
        preview_col, meta_col = st.columns([1.15, 1])
        with preview_col:
            st.video(input_path)
        with meta_col:
            m1, m2 = st.columns(2)
            m1.metric("Duration", f"{duration:.1f}s")
            m2.metric("Frame rate", f"{fps:.1f} FPS" if fps else "Unknown")
            m3, m4 = st.columns(2)
            m3.metric("Resolution", f"{width} × {height}" if width and height else "Unknown")
            m4.metric("File size", f"{uploaded.size / (1024 * 1024):.2f} MB")
            st.markdown(f'<div class="info-panel"><b>Current selection profile</b><br><span class="small-muted">{scoring_mode} · {segment_seconds}s segments · {sample_fps} sampled frames/sec</span></div>', unsafe_allow_html=True)
            if duration > 0 and summary_seconds > duration:
                st.info("The target is longer than this source video; the available duration will be used.")
            generate = st.button("🚀 Generate my short video", type="primary", use_container_width=True)
            if generate:
                progress = st.progress(0, text="Preparing video analysis...")
                try:
                    progress.progress(15, text="Extracting sampled frames...")
                    result = summarize_video(input_path, segment_seconds=segment_seconds, target_seconds=summary_seconds,
                                             sample_fps=sample_fps, output_dir="outputs", novelty_weight=novelty_weight)
                    progress.progress(100, text="Summary is ready!")
                    st.session_state["video_summary_result"] = result
                    st.session_state["video_summary_name"] = uploaded.name
                    st.session_state["video_summary_meta"] = {"duration": duration, "width": width, "height": height, "fps": fps, "size_mb": uploaded.size / (1024 * 1024), "style": scoring_mode}
                    st.success("Your video summary has been generated.")
                except Exception as exc:
                    progress.empty()
                    st.error(f"Could not summarize this video: {exc}")
                    st.info("Check that the dependencies are installed and try a shorter MP4 video. The first AI model run may take longer because weights need to load.")
    os.unlink(input_path)

if "video_summary_result" in st.session_state:
    result = st.session_state["video_summary_result"]
    meta = st.session_state.get("video_summary_meta", {})
    original = float(result.get("original_duration", 0))
    summary = float(result.get("summary_duration", 0))
    reduction = max(0.0, (1 - summary / original) * 100) if original > 0 else 0
    segments = result.get("segments", [])
    st.divider()
    st.markdown('<div class="eyebrow">YOUR GENERATED RESULT</div><div class="section-head">📌 Summary workspace</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_data = [("Original length", f"{original:.1f}s"), ("Summary length", f"{summary:.1f}s"), ("Time reduced", f"{reduction:.1f}%"), ("Highlights", str(len(segments)))]
    for col, (label, value) in zip(metric_cols, metric_data):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{st.session_state.get("video_summary_name", "Uploaded video")[:28]}</div></div>', unsafe_allow_html=True)

    overview_tab, summary_tab, highlights_tab, insights_tab, export_tab = st.tabs(["🏠 Overview", "▶️ Summary player", "🖼️ Highlights", "📊 Insights", "📦 Export centre"])
    with overview_tab:
        st.markdown("#### Summary at a glance")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.video(result["summary_path"])
        with c2:
            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
            st.markdown(f"**Source:** {st.session_state.get('video_summary_name', 'Uploaded video')}  ")
            st.markdown(f"**Selection style:** {meta.get('style', 'Balanced')}  ")
            st.markdown(f"**Scoring balance:** {result.get('novelty_weight', .8):.0%} visual novelty / {result.get('motion_weight', .2):.0%} motion  ")
            st.markdown(f"**Compression:** {reduction:.1f}% shorter than the source")
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("The selected clips are assembled in their original chronological order.")
    with summary_tab:
        st.markdown("#### Play your generated short video")
        st.video(result["summary_path"])
        with open(result["summary_path"], "rb") as vf:
            st.download_button("⬇️ Download MP4 summary", vf, file_name="clipsense_video_summary.mp4", mime="video/mp4", type="primary")
    with highlights_tab:
        st.markdown("#### Selected moments")
        if not segments:
            st.info("No highlight thumbnails are available.")
        else:
            for start in range(0, len(segments), 3):
                cols = st.columns(3)
                for col, item in zip(cols, segments[start:start + 3]):
                    with col:
                        st.image(item["thumbnail"], use_container_width=True)
                        st.markdown(f"**Highlight {start + segments[start:start + 3].index(item) + 1}** · {item['start']:.1f}s – {item['end']:.1f}s")
                        st.progress(min(1.0, max(0.0, float(item["score"]))), text=f"Score {item['score']:.3f}")
                        st.caption(f"Novelty {item['novelty']:.3f} · Motion {item['motion']:.3f}")
    with insights_tab:
        st.markdown("#### Highlight score analysis")
        df = pd.DataFrame([{"Highlight": f"H{i}", "Start (sec)": round(x["start"], 2), "End (sec)": round(x["end"], 2), "Duration (sec)": round(x["end"] - x["start"], 2), "Importance score": round(x["score"], 4), "Visual novelty": round(x["novelty"], 4), "Motion": round(x["motion"], 4)} for i, x in enumerate(segments, 1)])
        if not df.empty:
            chart_col, stat_col = st.columns([1.35, 1])
            with chart_col:
                st.bar_chart(df.set_index("Highlight")[["Importance score", "Visual novelty", "Motion"]])
            with stat_col:
                st.markdown('<div class="info-panel">', unsafe_allow_html=True)
                st.metric("Average highlight score", f"{df['Importance score'].mean():.3f}")
                st.metric("Longest selected segment", f"{df['Duration (sec)'].max():.1f}s")
                st.metric("Total selected time", f"{df['Duration (sec)'].sum():.1f}s")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("#### Detailed segment table")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("#### Timeline of selected clips")
            timeline = pd.DataFrame({"Start": [x["start"] for x in segments], "End": [x["end"] for x in segments], "Highlight": [f"H{i}" for i in range(1, len(segments) + 1)]})
            st.dataframe(timeline, use_container_width=True, hide_index=True)
        else:
            st.info("No analysis data available.")
    with export_tab:
        st.markdown("#### Download your files")
        rows = [{"Highlight": i, "Start (sec)": round(x["start"], 2), "End (sec)": round(x["end"], 2), "Duration (sec)": round(x["end"] - x["start"], 2), "Importance score": round(x["score"], 4), "Visual novelty": round(x["novelty"], 4), "Motion": round(x["motion"], 4)} for i, x in enumerate(segments, 1)]
        report_df = pd.DataFrame(rows)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("📄 Download CSV report", report_df.to_csv(index=False).encode("utf-8"), file_name="clipsense_highlight_report.csv", mime="text/csv", use_container_width=True)
        with e2:
            report = {"source_video": st.session_state.get("video_summary_name"), "original_duration_seconds": original, "summary_duration_seconds": summary, "time_reduced_percent": round(reduction, 2), "selection_style": meta.get("style", "Balanced"), "segments": rows}
            st.download_button("🧾 Download JSON report", json.dumps(report, indent=2).encode("utf-8"), file_name="clipsense_summary_report.json", mime="application/json", use_container_width=True)
        thumb_buffer = BytesIO()
        with zipfile.ZipFile(thumb_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, item in enumerate(segments, 1):
                if os.path.exists(item["thumbnail"]):
                    zf.write(item["thumbnail"], f"highlight_{i}.jpg")
        st.download_button("🖼️ Download all thumbnails (ZIP)", thumb_buffer.getvalue(), file_name="clipsense_highlight_thumbnails.zip", mime="application/zip", use_container_width=True)
        st.caption("Exports include the selected segment timings and calculated visual scores.")

st.divider()
st.caption("ClipSense AI · Educational project prototype · Highlight scores combine visual feature novelty and frame motion; human review is recommended.")
