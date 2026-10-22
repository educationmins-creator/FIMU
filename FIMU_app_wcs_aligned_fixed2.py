"""FIMU — professional astronomical FITS analysis workbench.

Run with: streamlit run app.py
Dependencies: streamlit, numpy, pandas, plotly, astropy, scipy (optional for source detection)
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch, SqrtStretch, ZScaleInterval

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE + DESIGN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FIMU | Astronomical Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#070b14; --panel:#0e1625; --line:#24344b; --muted:#8da2bf; --text:#edf4ff; --cyan:#67e8f9; --violet:#a78bfa; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #14233d 0%, var(--bg) 48%); color:var(--text); font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1728,#080d17); border-right:1px solid var(--line); }
.block-container { max-width:1600px; padding-top:1.5rem; padding-bottom:3rem; }
.hero { position:relative; overflow:hidden; padding:2rem 2.2rem; border:1px solid #2c4566; border-radius:24px; background:linear-gradient(115deg,rgba(24,52,88,.96),rgba(13,22,39,.94) 62%,rgba(39,29,70,.72)); box-shadow:0 20px 70px #0005; margin-bottom:1.2rem; }
.hero:after { content:'✦'; position:absolute; right:5%; top:-45px; font-size:190px; color:#8be9ff12; }
.eyebrow { font:500 .72rem 'DM Mono',monospace; letter-spacing:.18em; text-transform:uppercase; color:var(--cyan); }
.hero h1 { margin:.45rem 0 .35rem; font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; letter-spacing:-.055em; color:#f5f8ff; }
.hero p { max-width:760px; color:#b5c6df; font-size:1rem; margin:0; }
.section-head { font-size:1.12rem; font-weight:800; letter-spacing:-.025em; margin:.5rem 0 .2rem; }
.subtle { color:var(--muted); font-size:.86rem; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#111f33,#0c1422); border:1px solid #263a55; border-radius:16px; padding:14px 16px; }
div[data-testid="stMetricLabel"] { color:#9db1cd; }
div[data-testid="stMetricValue"] { color:#f0f7ff; }
.stButton button,.stDownloadButton button { border-radius:11px; border:1px solid #426a91; font-weight:700; }
.stButton button[kind="primary"] { background:linear-gradient(100deg,#0e7490,#6d4acb); border:0; }
[data-testid="stFileUploader"] { border:1px dashed #426486; border-radius:16px; background:#0c1524aa; padding:.6rem; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#101a2a; border:1px solid #263850; border-radius:10px 10px 0 0; padding:8px 14px; }
hr { border-color:#24344b; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=12)
def inspect_fits(raw: bytes):
    """Read a FITS upload once and return serializable HDU metadata and arrays."""
    result = []
    with fits.open(BytesIO(raw), memmap=False, do_not_scale_image_data=False) as hdul:
        for idx, hdu in enumerate(hdul):
            item = {
                "index": idx,
                "name": hdu.name or f"HDU {idx}",
                "type": type(hdu).__name__,
                "shape": tuple(hdu.data.shape) if hdu.data is not None and hasattr(hdu.data, "shape") else None,
                "header": dict(hdu.header),
                "is_image": isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and hdu.data is not None,
                "data": None,
            }
            if item["is_image"]:
                try:
                    arr = np.asarray(hdu.data, dtype=np.float32)
                    arr = np.squeeze(arr)
                    if arr.ndim == 2:
                        item["data"] = arr
                except (TypeError, ValueError, MemoryError):
                    pass
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.data is not None:
                item["columns"] = list(hdu.columns.names or [])
                if item["name"].upper() == "WCS-WAVE" and "VALUES" in item["columns"]:
                    try:
                        item["wave_values"] = np.asarray(hdu.data["VALUES"][0])
                        item["wave_x"] = np.asarray(hdu.data["X"][0], dtype=float) if "X" in item["columns"] else None
                        item["wave_y"] = np.asarray(hdu.data["Y"][0], dtype=float) if "Y" in item["columns"] else None
                    except (IndexError, TypeError, ValueError):
                        item["wave_values"] = None
                        item["wave_x"] = None
                        item["wave_y"] = None
            result.append(item)
    return result


def finite_stats(data: np.ndarray) -> dict | None:
    finite = np.isfinite(data)
    values = data[finite]
    if values.size == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    mad = float(1.4826 * np.median(np.abs(values - np.median(values))))
    return {
        "Pixels": int(data.size), "Valid pixels": int(values.size),
        "Invalid pixels": int(data.size - values.size),
        "Mean": float(np.mean(values)), "Median": float(np.median(values)),
        "Std. deviation": float(np.std(values)), "Sigma-clipped mean": float(mean),
        "Sigma-clipped median": float(median), "Sigma-clipped std": float(std),
        "Robust noise (MAD)": mad, "Minimum": float(np.min(values)),
        "Maximum": float(np.max(values)), "P01": float(q[0]), "P05": float(q[1]),
        "P25": float(q[2]), "P50": float(q[3]), "P75": float(q[4]),
        "P95": float(q[5]), "P99": float(q[6]),
    }


def make_display(data, stretch, low_pct, high_pct):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data, 0.0, 1.0
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    clipped = np.clip(data, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    if stretch == "Log":
        normalized = np.log1p(1000 * normalized) / np.log1p(1000)
    elif stretch == "Asinh":
        normalized = np.arcsinh(10 * normalized) / np.arcsinh(10)
    elif stretch == "Square root":
        normalized = np.sqrt(normalized)
    return np.where(np.isfinite(data), normalized, np.nan), float(lo), float(hi)


def wcs_header_only(header):
    """Build a clean Astropy Header containing WCS cards only.

    FITS COMMENT/HISTORY cards can contain multiline or non-ASCII text. Passing
    a plain dict of every header value into WCS may make Astropy parse that text
    as if it were a FITS header. Keep only standard coordinate-related cards.
    """
    clean = fits.Header()
    key_pattern = re.compile(
        r"^(WCSAXES|CTYPE\d+[A-Z]?|CUNIT\d+[A-Z]?|CRPIX\d+[A-Z]?|"
        r"CRVAL\d+[A-Z]?|CDELT\d+[A-Z]?|CROTA\d+[A-Z]?|"
        r"PC\d+_\d+[A-Z]?|CD\d+_\d+[A-Z]?|"
        r"PV\d+_\d+[A-Z]?|PS\d+_\d+[A-Z]?|"
        r"RADESYS[A-Z]?|RADECSYS|EQUINOX|LONPOLE|LATPOLE|WCSNAME[A-Z]?|"
        r"MJDREF|MJDREFI|MJDREFF|RESTFRQ|RESTWAV|SPECSYS|SSYSOBS|VELOSYS|"
        r"A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_DMAX|B_DMAX|AP_DMAX|BP_DMAX|"
        r"A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+)$"
    )
    for key, value in header.items():
        if not isinstance(key, str) or not key_pattern.match(key.upper()):
            continue
        try:
            clean[key] = value
        except (ValueError, TypeError, UnicodeError):
            # Ignore malformed WCS cards; WCS will then report missing/incomplete
            # coordinate metadata instead of choking on unrelated COMMENT text.
            continue
    return clean


def celestial_wcs_from_header(header):
    """Parse celestial WCS while excluding arbitrary COMMENT/HISTORY text."""
    return WCS(wcs_header_only(header), relax=True).celestial


def wcs_summary(header):
    try:
        wcs = WCS(wcs_header_only(header), relax=True)
        if wcs.has_celestial and wcs.pixel_n_dim >= 2:
            return "Celestial WCS detected", wcs
    except Exception:
        pass
    return "No valid celestial WCS detected", None


def detect_sources(data, threshold_sigma=5.0, min_pixels=3):
    if not SCIPY_AVAILABLE:
        return None, "Install scipy to enable connected-component source detection."
    finite = np.isfinite(data)
    if finite.sum() < 10:
        return pd.DataFrame(), None
    _, med, _ = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
    noise = 1.4826 * np.median(np.abs(data[finite] - np.median(data[finite])))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(data[finite]))
    if not np.isfinite(noise) or noise <= 0:
        return pd.DataFrame(), None
    mask = finite & (data > med + threshold_sigma * noise)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    rows = []
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        region = labels[slc] == label_id
        yy, xx = np.nonzero(region)
        if len(xx) < min_pixels:
            continue
        yy = yy + slc[0].start
        xx = xx + slc[1].start
        vals = data[yy, xx]
        weights = np.maximum(vals - med, 0)
        if weights.sum() > 0:
            cx = float(np.average(xx, weights=weights)); cy = float(np.average(yy, weights=weights))
        else:
            cx = float(np.mean(xx)); cy = float(np.mean(yy))
        rows.append({"Source": len(rows) + 1, "X (pixel)": cx, "Y (pixel)": cy,
                     "Peak signal": float(np.max(vals)), "Integrated excess": float(np.sum(vals - med)),
                     "Area (pixels)": int(len(xx))})
    return pd.DataFrame(rows), None


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}"
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)

# -----------------------------------------------------------------------------
# SIDEBAR + HEADER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# ✦ FIMU")
    st.caption("ASTRONOMICAL INTELLIGENCE PLATFORM")
    st.markdown("---")
    st.markdown("**WORKSPACE**")
    st.markdown("🛰️  FITS image analysis  \n🌈  Wavelength calibration  \n📊  Observation comparison  \n🎯  Candidate source detection")
    st.markdown("---")
    st.markdown("**ANALYSIS NOTES**")
    st.caption("Statistics use the full-resolution image. Display contrast only changes the visualization.")
    st.caption("Source detection is a simple threshold-based candidate finder, not a validated scientific catalogue.")
    st.markdown("---")
    st.caption("FIMU • Explore the universe with evidence.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">FIMU / SPACE SCIENCE WORKBENCH</div>
  <h1>See deeper. Measure smarter.</h1>
  <p>A premium workspace for inspecting astronomical FITS images, auditing metadata, exploring wavelength grids, and comparing observations with reproducible measurements.</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Drop your FITS observations here", type=["fits", "fit", "fts"], accept_multiple_files=True,
                                  help="Upload one or more .fits, .fit, or .fts files. Large files are analyzed in memory.")

if not uploaded_files:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><div class='eyebrow'>01 / VISUALIZE</div><h3>Image intelligence</h3><p class='subtle'>Percentile controls, linear/log/asinh/sqrt stretches, and intensity distribution.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='eyebrow'>02 / QUANTIFY</div><h3>Robust statistics</h3><p class='subtle'>Sigma-clipped estimates, robust noise, invalid-pixel audit, and exportable reports.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='eyebrow'>03 / COMPARE</div><h3>Observation lab</h3><p class='subtle'>Compare image metrics and compatible wavelength calibration grids.</p></div>", unsafe_allow_html=True)
    st.info("Upload FITS files to begin. Your original files are not modified.")
    st.stop()

# Read each upload once through Streamlit's data cache.
observations = []
for upload in uploaded_files:
    try:
        hdus = inspect_fits(upload.getvalue())
        image_choices = [h for h in hdus if h.get("data") is not None]
        if not image_choices:
            st.warning(f"{upload.name}: no two-dimensional image HDU was found.")
            continue
        observations.append({"name": upload.name, "digest": hashlib.sha1(upload.getvalue()).hexdigest()[:10], "hdus": hdus, "images": image_choices})
    except Exception as exc:
        st.error(f"Could not read **{upload.name}**: {exc}")

if not observations:
    st.stop()

st.markdown("<div class='section-head'>Observation workspace</div><div class='subtle'>Select an image extension per file. Full-resolution measurements are kept separate from display scaling.</div>", unsafe_allow_html=True)
file_tabs = st.tabs([f"✦ {o['name'][:22]}" for o in observations])
summary_rows = []
wavelength_grids = []

for obs, tab in zip(observations, file_tabs):
    with tab:
        image_names = [f"HDU {h['index']} · {h['name']} · {h['data'].shape[1]}×{h['data'].shape[0]}" for h in obs["images"]]
        chosen_label = st.selectbox("Image extension", image_names, key=f"hdu_{obs['digest']}")
        chosen = obs["images"][image_names.index(chosen_label)]
        data = chosen["data"]
        header = chosen["header"]
        stats = finite_stats(data)
        if stats is None:
            st.error("This image has no finite pixels.")
            continue
        unit = str(header.get("BUNIT", "Unit not specified"))
        exposure = header.get("EXPTIME", header.get("EXPOSURE", "Not recorded"))
        object_name = header.get("OBJECT", "Object not recorded")
        date_obs = header.get("DATE-OBS", "Date not recorded")
        summary_rows.append({"File": obs["name"], "HDU": chosen["index"], "Object": object_name, "Date observed": date_obs,
                             "Exposure": exposure, "Unit": unit, **stats})
        st.markdown(f"### {object_name if object_name != 'Object not recorded' else obs['name']}")
        st.caption(f"{obs['name']} · HDU {chosen['index']} ({chosen['name']}) · {data.shape[1]:,} × {data.shape[0]:,} pixels · {unit}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median signal", fmt(stats["Median"]))
        m2.metric("Robust noise (MAD)", fmt(stats["Robust noise (MAD)"]))
        m3.metric("Valid pixels", fmt(stats["Valid pixels"]))
        m4.metric("Invalid pixels", fmt(stats["Invalid pixels"]))

        image_tab, stats_tab, meta_tab, source_tab, wave_tab = st.tabs(["🖼️ Image lab", "📈 Statistics", "🧾 FITS metadata", "🎯 Source candidates", "🌈 Wavelength"])
        with image_tab:
            left, right = st.columns([1, 3])
            with left:
                stretch = st.selectbox("Intensity stretch", ["Linear", "Log", "Asinh", "Square root"], index=2, key=f"stretch_{obs['digest']}_{chosen['index']}")
                pct = st.slider("Display percentile range", 0, 100, (1, 99), key=f"pct_{obs['digest']}_{chosen['index']}")
                cmap = st.selectbox("Color map", ["Turbo", "Viridis", "Cividis", "Magma", "Inferno", "Gray"], index=3, key=f"cmap_{obs['digest']}_{chosen['index']}")
                st.caption("Visualization controls never alter measured pixel values.")
                st.markdown("**Header quick facts**")
                st.write(f"Exposure: `{exposure}`")
                st.write(f"Observed: `{date_obs}`")
                wcs_status, celestial_wcs = wcs_summary(header)
                st.write(f"WCS: `{wcs_status}`")
            with right:
                display, lo, hi = make_display(data, stretch, pct[0], pct[1])
                # Keep browser payload manageable for very large frames; measurements remain full resolution.
                max_side = 1800
                step = max(1, int(np.ceil(max(data.shape) / max_side)))
                view = display[::step, ::step]
                fig = px.imshow(view, origin="upper", color_continuous_scale=cmap, aspect="equal",
                                labels={"x": "X pixel", "y": "Y pixel", "color": "Normalized intensity"},
                                title=f"{stretch} stretch · display limits {lo:.4g} to {hi:.4g}")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=640, margin=dict(l=8, r=8, t=52, b=8), font=dict(family="Manrope"))
                fig.update_xaxes(title="X pixel (display grid)")
                fig.update_yaxes(title="Y pixel (display grid)")
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
                if step > 1:
                    st.caption(f"Large image display sampled every {step} pixels for browser performance; all statistics use all {data.size:,} pixels.")
            finite = data[np.isfinite(data)]
            hist = px.histogram(x=finite[::max(1, finite.size // 250000)], nbins=100, title="Pixel intensity distribution", labels={"x": unit, "count": "Pixel count"}, template="plotly_dark")
            hist.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(hist, use_container_width=True)
        with stats_tab:
            st.markdown("**Full-resolution measurements**")
            stats_df = pd.DataFrame([{"Statistic": k, "Value": v} for k, v in stats.items()])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.download_button("Download statistics CSV", stats_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{obs['name']}_statistics.csv", mime="text/csv", key=f"stats_csv_{obs['digest']}_{chosen['index']}")
            st.caption("Sigma-clipped values use a 3σ clipping threshold. MAD noise is 1.4826 × median absolute deviation; neither metric guarantees a pure background estimate.")
        with meta_tab:
            rows = [{"Keyword": key, "Value": str(value)} for key, value in header.items() if key and key not in ("COMMENT", "HISTORY")]
            metadata_df = pd.DataFrame(rows)
            search_meta = st.text_input("Filter metadata keywords or values", key=f"metasearch_{obs['digest']}_{chosen['index']}")
            if search_meta:
                mask = metadata_df.astype(str).apply(lambda col: col.str.contains(search_meta, case=False, regex=False)).any(axis=1)
                metadata_df = metadata_df[mask]
            st.dataframe(metadata_df, use_container_width=True, height=420, hide_index=True)
            st.download_button("Download complete header CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                               file_name=f"{obs['name']}_metadata.csv", mime="text/csv", key=f"meta_csv_{obs['digest']}_{chosen['index']}")
            st.markdown("**Coordinate reference**")
            st.write(wcs_status)
            if celestial_wcs is not None:
                try:
                    center = celestial_wcs.pixel_to_world((data.shape[1]-1)/2, (data.shape[0]-1)/2)
                    st.write(f"Image center: RA `{center.ra.deg:.7f}°`, Dec `{center.dec.deg:.7f}°`")
                except Exception:
                    st.caption("WCS exists, but center coordinates could not be calculated.")
        with source_tab:
            threshold = st.slider("Detection threshold (robust σ)", 2.0, 10.0, 5.0, 0.5, key=f"thresh_{obs['digest']}_{chosen['index']}")
            min_area = st.slider("Minimum connected pixels", 1, 30, 3, key=f"area_{obs['digest']}_{chosen['index']}")
            if st.button("Find source candidates", type="primary", key=f"detect_{obs['digest']}_{chosen['index']}"):
                with st.spinner("Scanning full-resolution image…"):
                    found, error = detect_sources(data, threshold, min_area)
                if error:
                    st.warning(error)
                else:
                    st.session_state[f"sources_{obs['digest']}_{chosen['index']}"] = found
                    st.session_state[f"sources_cfg_{obs['digest']}_{chosen['index']}"] = (threshold, min_area)
            found = st.session_state.get(f"sources_{obs['digest']}_{chosen['index']}")
            if found is not None:
                st.metric("Candidate sources", len(found))
                st.dataframe(found, use_container_width=True, hide_index=True)
                st.download_button("Download candidate catalogue CSV", found.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{obs['name']}_source_candidates.csv", mime="text/csv", key=f"sources_csv_{obs['digest']}_{chosen['index']}")
                st.caption("Thresholding can miss faint objects and can flag cosmic rays, hot pixels, or noise. Validate candidates before scientific use.")
            else:
                st.info("Run candidate detection to identify connected groups of pixels above the selected robust-noise threshold.")
        with wave_tab:
            wave_hdu = next((h for h in obs["hdus"] if h["name"].upper() == "WCS-WAVE"), None)
            if not wave_hdu:
                st.info("No WCS-WAVE extension was found in this FITS file.")
            elif wave_hdu.get("wave_values") is None:
                st.warning("WCS-WAVE exists, but its VALUES column could not be read.")
            else:
                wave = np.asarray(wave_hdu["wave_values"])
                st.write(f"VALUES array shape: `{wave.shape}`")
                if wave.ndim == 3 and wave.shape[-1] >= 1:
                    grid = wave[:, :, 0]
                    wavelength_grids.append({
                        "file": obs["name"],
                        "grid": grid,
                        "x_coords": wave_hdu.get("wave_x"),
                        "y_coords": wave_hdu.get("wave_y"),
                        "image_header": header,
                    })
                    wf = px.imshow(grid, origin="upper", color_continuous_scale="Turbo", title="Wavelength calibration grid",
                                   labels={"x": "Grid X", "y": "Grid Y", "color": "Wavelength (header-defined units)"})
                    wf.update_layout(template="plotly_dark", height=480, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(wf, use_container_width=True)
                    st.caption("Confirm wavelength units and axis conventions in the instrument documentation before interpreting values.")
                else:
                    st.warning("VALUES does not match the expected 3D layout with wavelength values in the last axis.")
                colnames = wave_hdu.get("columns", [])
                st.write("Table columns:", ", ".join(colnames) if colnames else "No table columns found")
                if colnames:
                    st.dataframe(pd.DataFrame([{"Column": name, "Shape": str(np.shape(wave_hdu.get('wave_values') if name == 'VALUES' else ''))} for name in colnames]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# CROSS-OBSERVATION ANALYSIS
# -----------------------------------------------------------------------------
if len(summary_rows) > 1:
    st.markdown("---")
    st.markdown("<div class='section-head'>Cross-observation analysis</div><div class='subtle'>Compare selected image extensions. Interpret differences in context: exposure, units, calibration, and instrument setup may differ.</div>", unsafe_allow_html=True)
    comparison = pd.DataFrame(summary_rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    metrics = ["Mean", "Median", "Std. deviation", "Sigma-clipped mean", "Sigma-clipped std", "Robust noise (MAD)", "Minimum", "Maximum", "Valid pixels", "Invalid pixels"]
    chosen_metric = st.selectbox("Metric to compare", metrics)
    chart = px.bar(comparison, x="File", y=chosen_metric, color="File", title=f"{chosen_metric} by observation", template="plotly_dark", hover_data=["HDU", "Object", "Date observed", "Unit"])
    chart.update_layout(height=400, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, use_container_width=True)
    st.download_button("Download combined observation report", comparison.to_csv(index=False).encode("utf-8"), file_name="FIMU_observation_comparison.csv", mime="text/csv")

if len(wavelength_grids) >= 2:
    st.markdown("---")
    st.markdown("<div class='section-head'>Wavelength-grid difference</div>", unsafe_allow_html=True)
    labels = [f"{g['file']} · grid {i+1}" for i, g in enumerate(wavelength_grids)]
    a_label = st.selectbox("Reference grid", labels, index=0, key="wave_ref")
    b_options = [x for x in labels if x != a_label]
    b_label = st.selectbox("Comparison grid", b_options, index=0, key="wave_cmp")
    ref = wavelength_grids[labels.index(a_label)]
    cmp = wavelength_grids[labels.index(b_label)]
    a = np.asarray(ref["grid"], dtype=float)
    b = np.asarray(cmp["grid"], dtype=float)

    st.caption(
        "The WCS-WAVE table stores wavelength samples at detector X/Y positions; "
        "celestial coordinates are taken from each selected IMAGE extension header."
    )
    if st.button("Align grids using image celestial WCS", type="primary", key="align_wave_wcs"):
        try:
            from scipy.interpolate import griddata

            ref_wcs = celestial_wcs_from_header(ref["image_header"])
            cmp_wcs = celestial_wcs_from_header(cmp["image_header"])
            if not ref_wcs.has_celestial or not cmp_wcs.has_celestial:
                raise ValueError("The selected IMAGE headers do not both define celestial WCS.")
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy is required for coordinate-based interpolation. Install it with: pip install scipy")

            ref_x = np.asarray(ref["x_coords"], dtype=float)
            ref_y = np.asarray(ref["y_coords"], dtype=float)
            cmp_x = np.asarray(cmp["x_coords"], dtype=float)
            cmp_y = np.asarray(cmp["y_coords"], dtype=float)
            if a.shape != (len(ref_y), len(ref_x)):
                raise ValueError(f"Reference grid shape {a.shape} does not match its X/Y coordinate lengths ({len(ref_y)}, {len(ref_x)}).")
            if b.shape != (len(cmp_y), len(cmp_x)):
                raise ValueError(f"Comparison grid shape {b.shape} does not match its X/Y coordinate lengths ({len(cmp_y)}, {len(cmp_x)}).")

            # WCS pixel coordinates are zero-based; FITS WCS-WAVE X/Y samples are one-based.
            rxx, ryy = np.meshgrid(ref_x - 1.0, ref_y - 1.0)
            cxx, cyy = np.meshgrid(cmp_x - 1.0, cmp_y - 1.0)
            ref_ra, ref_dec = ref_wcs.pixel_to_world_values(rxx, ryy)
            cmp_ra, cmp_dec = cmp_wcs.pixel_to_world_values(cxx, cyy)

            # Use a local tangent approximation around the reference field centre;
            # wrap RA differences to avoid a discontinuity at 0/360 degrees.
            finite_a = np.isfinite(a) & np.isfinite(ref_ra) & np.isfinite(ref_dec)
            finite_b = np.isfinite(b) & np.isfinite(cmp_ra) & np.isfinite(cmp_dec)
            if finite_a.sum() < 3 or finite_b.sum() < 3:
                raise ValueError("Too few finite wavelength samples with valid celestial coordinates.")
            ra0 = float(np.nanmedian(ref_ra[finite_a]))
            dec0 = float(np.nanmedian(ref_dec[finite_a]))
            def tangent_xy(ra, dec):
                dra = (ra - ra0 + 180.0) % 360.0 - 180.0
                return np.column_stack(((dra * np.cos(np.deg2rad(dec0))).ravel(), (dec - dec0).ravel()))

            ref_points = tangent_xy(ref_ra, ref_dec)
            cmp_points = tangent_xy(cmp_ra[finite_b], cmp_dec[finite_b])
            cmp_values = b[finite_b]
            query_points = ref_points
            aligned_flat = griddata(cmp_points, cmp_values, query_points, method="linear")
            aligned_b = aligned_flat.reshape(a.shape)
            aligned_b[~finite_a] = np.nan

            difference = aligned_b - a
            finite_diff = difference[np.isfinite(difference)]
            if finite_diff.size == 0:
                raise ValueError("The two grids have no overlapping area with valid interpolated samples.")
            st.success(
                f"Aligned by celestial coordinates from the IMAGE headers. "
                f"Valid overlap: {finite_diff.size} of {a.size} reference grid points."
            )
            d1, d2, d3 = st.columns(3)
            d1.metric("Mean difference", fmt(np.mean(finite_diff)))
            d2.metric("Median difference", fmt(np.median(finite_diff)))
            d3.metric("Max absolute difference", fmt(np.max(np.abs(finite_diff))))
            df = px.imshow(
                difference, origin="upper", color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title="Wavelength difference (comparison − reference)",
                labels={"x": "Reference grid X", "y": "Reference grid Y", "color": "Difference (table units)"},
            )
            df.update_layout(template="plotly_dark", height=480, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(df, use_container_width=True)
            st.download_button(
                "Download wavelength difference grid",
                pd.DataFrame(difference).to_csv(index=False).encode("utf-8"),
                file_name="FIMU_wavelength_difference.csv",
                mime="text/csv",
            )
            st.caption(
                "Linear interpolation only reports the overlapping sky footprint. "
                "Validate the wavelength/bandpass column meaning and units against instrument documentation."
            )
        except Exception as exc:
            st.error(f"Coordinate-based wavelength alignment could not be performed: {exc}")
            st.info(
                "No difference map was calculated. This method uses celestial WCS from the selected IMAGE headers, "
                "not from WCS-WAVE, because these table extensions contain X/Y sample positions and VALUES."
            )

st.markdown("---")
st.caption("FIMU · Astronomical Intelligence Platform · Analysis outputs are exploratory and should be validated against instrument documentation and scientific workflows.")
