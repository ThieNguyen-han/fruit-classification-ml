"""
app.py - FRUITVISION AI
Hệ thống phân loại trái cây thông minh bằng SVM và Random Forest
Ứng dụng web Streamlit đa trang với phong cách AI Dashboard & Scientific Visualization.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import module trích xuất đặc trưng từ src
from src.features import extract_hog_features, preprocess_image


# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="FRUITVISION AI | Phân loại trái cây thông minh",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ĐỊNH NGHĨA ĐƯỜNG DẪN & HẰNG SỐ
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ORIG_DATA_DIR = DATA_DIR / "Original Image"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
LABELS_PATH = MODELS_DIR / "labels.joblib"
METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CM_SVM_PATH = RESULTS_DIR / "confusion_matrix_svm.png"
CM_RF_PATH = RESULTS_DIR / "confusion_matrix_rf.png"
COMP_CHART_PATH = RESULTS_DIR / "model_comparison.png"

# Bảng thông tin chuẩn hóa cho 16 lớp trái cây (Song ngữ Anh - Việt)
CLASS_METADATA = {
    "FreshApple": {"vi": "Táo tươi", "fruit": "Táo", "status": "Tươi", "is_fresh": True},
    "FreshBanana": {"vi": "Chuối tươi", "fruit": "Chuối", "status": "Tươi", "is_fresh": True},
    "FreshGrape": {"vi": "Nho tươi", "fruit": "Nho", "status": "Tươi", "is_fresh": True},
    "FreshGuava": {"vi": "Ổi tươi", "fruit": "Ổi", "status": "Tươi", "is_fresh": True},
    "FreshJujube": {"vi": "Táo ta tươi", "fruit": "Táo ta (Táo tàu)", "status": "Tươi", "is_fresh": True},
    "FreshOrange": {"vi": "Cam tươi", "fruit": "Cam", "status": "Tươi", "is_fresh": True},
    "FreshPomegranate": {"vi": "Lựu tươi", "fruit": "Lựu", "status": "Tươi", "is_fresh": True},
    "FreshStrawberry": {"vi": "Dâu tây tươi", "fruit": "Dâu tây", "status": "Tươi", "is_fresh": True},
    "RottenApple": {"vi": "Táo hỏng", "fruit": "Táo", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenBanana": {"vi": "Chuối hỏng", "fruit": "Chuối", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenGrape": {"vi": "Nho hỏng", "fruit": "Nho", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenGuava": {"vi": "Ổi hỏng", "fruit": "Ổi", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenJujube": {"vi": "Táo ta hỏng", "fruit": "Táo ta (Táo tàu)", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenOrange": {"vi": "Cam hỏng", "fruit": "Cam", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenPomegranate": {"vi": "Lựu hỏng", "fruit": "Lựu", "status": "Hỏng / Thối", "is_fresh": False},
    "RottenStrawberry": {"vi": "Dâu tây hỏng", "fruit": "Dâu tây", "status": "Hỏng / Thối", "is_fresh": False},
}


# ==============================================================================
# DESIGN SYSTEM: CUSTOM CSS (AI DASHBOARD & SCIENTIFIC VISUALIZATION)
# ==============================================================================
st.markdown("""
<style>
    /* Nhúng font Inter thanh lịch */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    /* Tối ưu nền chính */
    .stApp {
        background-color: #F8FAF9;
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 0 16px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #1B4332;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0;
        font-weight: 500;
    }

    /* Page Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        color: #FFFFFF;
        padding: 2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(27, 67, 50, 0.2);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #D8F3DC;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
        color: #FFFFFF;
    }
    .hero-desc {
        font-size: 1.02rem;
        color: #D8F3DC;
        margin: 0;
        max-width: 820px;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1B4332;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #059669;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Section Cards */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    .card-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Model Prediction Cards */
    .model-card-svm {
        background: #FFFFFF;
        border: 1px solid #BFDBFE;
        border-top: 4px solid #2563EB;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.05);
    }
    .model-card-rf {
        background: #FFFFFF;
        border: 1px solid #BBF7D0;
        border-top: 4px solid #16A34A;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.05);
    }
    .model-tag {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }
    .model-pred-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0.3rem 0;
    }

    /* Badges */
    .badge-fresh {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-rotten {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Consensus Banners */
    .consensus-agree {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #166534;
        margin: 1.5rem 0;
    }
    .consensus-disagree {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #92400E;
        margin: 1.5rem 0;
    }

    /* Sidebar Status Block */
    .status-block {
        background: #F1F5F9;
        border-radius: 10px;
        padding: 12px;
        margin-top: 20px;
        font-size: 0.82rem;
    }
    .status-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        color: #475569;
    }
    .status-val {
        font-weight: 600;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HÀM PHỤ TRỢ: LOAD MODEL & DATA
# ==============================================================================
@st.cache_resource
def load_ml_models():
    """Tải hai mô hình SVM, Random Forest và danh sách nhãn đã huấn luyện."""
    if not (SVM_MODEL_PATH.exists() and RF_MODEL_PATH.exists() and LABELS_PATH.exists()):
        return None, None, None
    try:
        svm = joblib.load(SVM_MODEL_PATH)
        rf = joblib.load(RF_MODEL_PATH)
        labels = joblib.load(LABELS_PATH)
        return svm, rf, labels
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp mô hình đã lưu: {e}")
        return None, None, None


@st.cache_data
def load_metrics_data():
    """Tải số liệu đánh giá thực nghiệm từ tệp JSON."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_label_info(label_name: str) -> Dict:
    """Tra cứu thông tin hiển thị song ngữ Anh - Việt cho nhãn."""
    return CLASS_METADATA.get(
        label_name,
        {
            "vi": label_name,
            "fruit": label_name,
            "status": "Không xác định",
            "is_fresh": "Fresh" in label_name
        }
    )


# ==============================================================================
# KHỞI TẠO SESSION STATE
# ==============================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "📌 Tổng quan"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==============================================================================
# GIAO DIỆN THANH ĐIỀU HƯỚNG (SIDEBAR)
# ==============================================================================
def render_sidebar(svm_model, rf_model, labels, metrics_data):
    with st.sidebar:
        # Brand Logo & Tagline
        st.markdown("""
        <div class="brand-container">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5"></path>
                <path d="M8.5 8.5v.01"></path>
                <path d="M16 16v.01"></path>
                <path d="M12 12v.01"></path>
            </svg>
            <div>
                <h1 class="brand-title">FRUITVISION AI</h1>
                <p class="brand-subtitle">Hệ thống phân loại trái cây thông minh</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("ĐIỀU HƯỚNG HỆ THỐNG")

        pages = [
            "📌 Tổng quan",
            "🔍 Phân loại trái cây",
            "📊 So sánh mô hình",
            "📁 Dữ liệu và đánh giá",
            "📖 Giới thiệu đề tài"
        ]

        # Đồng bộ lựa chọn menu với session_state
        selected_page = st.radio(
            "Chọn trang chức năng:",
            pages,
            index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
            label_visibility="collapsed"
        )
        st.session_state.current_page = selected_page

        st.markdown("---")
        st.caption("TRẠNG THÁI HỆ THỐNG")

        models_ready = svm_model is not None and rf_model is not None and labels is not None
        data_ready = ORIG_DATA_DIR.exists() and len(list(ORIG_DATA_DIR.glob("*"))) > 0

        status_model_text = '<span style="color:#16A34A;font-weight:700;">● Đã sẵn sàng</span>' if models_ready else '<span style="color:#DC2626;font-weight:700;">● Chưa nạp</span>'
        status_data_text = '<span style="color:#16A34A;font-weight:700;">● 3,200 ảnh gốc</span>' if data_ready else '<span style="color:#DC2626;font-weight:700;">● Chưa tìm thấy</span>'

        st.markdown(f"""
        <div class="status-block">
            <div class="status-item">
                <span>Mô hình AI:</span>
                <span class="status-val">{status_model_text}</span>
            </div>
            <div class="status-item">
                <span>Tập dữ liệu:</span>
                <span class="status-val">{status_data_text}</span>
            </div>
            <div class="status-item">
                <span>Số lớp nhận diện:</span>
                <span class="status-val">{len(labels) if labels else 0} Lớp</span>
            </div>
            <div class="status-item">
                <span>Bộ xử lý:</span>
                <span class="status-val">CPU Optimized</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Đề tài môn Học máy (Machine Learning)")


# ==============================================================================
# TRANG 1: TỔNG QUAN (DASHBOARD)
# ==============================================================================
def render_overview_page(metrics_data, labels):
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            Dashboard Nghiên cứu & Thực nghiệm
        </div>
        <h1 class="hero-title">Phân loại trái cây bằng Machine Learning</h1>
        <p class="hero-desc">Khám phá và so sánh hiệu năng phân loại giữa hai thuật toán Support Vector Machine (SVM) và Random Forest trên không gian đặc trưng Histogram of Oriented Gradients (HOG).</p>
    </div>
    """, unsafe_allow_html=True)

    # 4 KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_classes = len(labels) if labels else (len(metrics_data.get("classes", [])) if metrics_data else 0)
    total_samples = metrics_data.get("dataset_summary", {}).get("total_samples", 0) if metrics_data else 0
    svm_acc = (metrics_data.get("svm", {}).get("accuracy", 0) * 100) if metrics_data else 0
    rf_acc = (metrics_data.get("random_forest", {}).get("accuracy", 0) * 100) if metrics_data else 0

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Số loại trái cây</div>
            <div class="kpi-value">{total_classes if total_classes > 0 else '—'}</div>
            <div class="kpi-subtext">8 Quả tươi + 8 Quả hỏng</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Tập dữ liệu gốc</div>
            <div class="kpi-value">{total_samples if total_samples > 0 else '—'}</div>
            <div class="kpi-subtext">200 mẫu cân bằng/lớp</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #2563EB;">
            <div class="kpi-label">SVM Accuracy</div>
            <div class="kpi-value" style="color:#2563EB;">{f"{svm_acc:.2f}%" if svm_acc > 0 else '—'}</div>
            <div class="kpi-subtext" style="color:#1D4ED8;">RBF Kernel (C=10.0)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #16A34A;">
            <div class="kpi-label">Random Forest Accuracy</div>
            <div class="kpi-value" style="color:#16A34A;">{f"{rf_acc:.2f}%" if rf_acc > 0 else '—'}</div>
            <div class="kpi-subtext" style="color:#15803D;">100 Decision Trees</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Biểu đồ so sánh và tóm tắt thực nghiệm
    col_chart, col_summary = st.columns([1.3, 1])

    with col_chart:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-heading">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                So sánh các chỉ số hiệu năng cốt lõi (%)
            </div>
        """, unsafe_allow_html=True)

        if metrics_data:
            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
            svm_vals = [
                metrics_data["svm"]["accuracy"] * 100,
                metrics_data["svm"]["precision_macro"] * 100,
                metrics_data["svm"]["recall_macro"] * 100,
                metrics_data["svm"]["f1_macro"] * 100
            ]
            rf_vals = [
                metrics_data["random_forest"]["accuracy"] * 100,
                metrics_data["random_forest"]["precision_macro"] * 100,
                metrics_data["random_forest"]["recall_macro"] * 100,
                metrics_data["random_forest"]["f1_macro"] * 100
            ]

            fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor='#FFFFFF')
            x = np.arange(len(metric_names))
            width = 0.32

            rects1 = ax.bar(x - width/2, svm_vals, width, label="SVM (RBF Kernel)", color="#2563EB", edgecolor="none", zorder=3)
            rects2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest", color="#10B981", edgecolor="none", zorder=3)

            ax.set_ylabel("Điểm số (%)", fontsize=10, fontweight="600", color="#475569")
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, fontsize=10, fontweight="600", color="#1E293B")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", linestyle="--", alpha=0.35, color="#94A3B8", zorder=0)
            ax.legend(frameon=False, loc="upper right", fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#CBD5E1")
            ax.spines['bottom'].set_color("#CBD5E1")

            # Gán nhãn giá trị
            for r in rects1:
                h = r.get_height()
                ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="700", color="#1D4ED8")
            for r in rects2:
                h = r.get_height()
                ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="700", color="#047857")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Chưa có số liệu thực nghiệm. Vui lòng chạy script huấn luyện `train_model.py`.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_summary:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-heading">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                Tóm tắt kết quả thực nghiệm
            </div>
            <div style="font-size:0.92rem; line-height:1.6; color:#334155;">
                <p><b>1. Độ chính xác tổng quan:</b><br>
                Mô hình <b>SVM</b> đạt độ chính xác <b>71.41%</b>, vượt trội hơn Random Forest (<b>62.34%</b>) chênh lệch <b>+9.07 điểm phần trăm</b> trên 16 lớp phân loại phân tầng.</p>
                <p><b>2. Đặc trưng HOG (1,764 chiều):</b><br>
                Đặc trưng HOG mô tả biên dạng hình học và mật độ gradient. Không gian chiều cao này rất phù hợp với siêu phẳng phi tuyến của RBF Kernel.</p>
                <p><b>3. Tốc độ tính toán:</b><br>
                Random Forest huấn luyện nhanh hơn SVM khoảng <b>18 lần</b> (1.57s vs 28.29s). Cả hai mô hình đều có thời gian suy luận dưới <b>3 mili-giây</b> mỗi ảnh trên CPU.</p>
            </div>
            <div style="margin-top: 1.5rem; text-align: center;">
        """, unsafe_allow_html=True)

        if st.button("🚀 Thử phân loại ảnh ngay →", use_container_width=True, type="primary"):
            st.session_state.current_page = "🔍 Phân loại trái cây"
            st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)


# ==============================================================================
# TRANG 2: PHÂN LOẠI TRÁI CÂY (CORE FEATURE)
# ==============================================================================
def render_classification_page(svm_model, rf_model, labels):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 800; color: #1E293B; margin: 0 0 0.4rem 0;">Phân loại và đối chiếu dự đoán</h2>
        <p style="color: #64748B; margin: 0; font-size: 0.95rem;">Tải lên hình ảnh trái cây để kiểm tra và đối chiếu kết quả dự đoán song song giữa hai mô hình SVM và Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    if svm_model is None or rf_model is None:
        st.warning("⚠️ Chưa phát hiện mô hình đã huấn luyện trong thư mục `models/`.")
        st.info("Vui lòng mở Terminal và thực thi lệnh: `python train_model.py` để huấn luyện mô hình trước khi thử nghiệm.")
        return

    # Khu vực Tải ảnh & Chọn ảnh mẫu
    col_upload, col_sample = st.columns([1.4, 1])

    uploaded_file = None
    sample_image_path = None

    with col_upload:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-heading">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                Tải ảnh từ máy tính
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Chọn file ảnh định dạng JPG, JPEG, PNG hoặc WEBP:",
            type=["jpg", "jpeg", "png", "webp"],
            help="Nên dùng ảnh quả chụp trên phông nền rõ ràng để đặc trưng biên HOG đạt độ chuẩn xác cao."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sample:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-heading">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>
                Thử nhanh với ảnh mẫu có sẵn
            </div>
            <p style="font-size:0.85rem; color:#64748B; margin-top:-0.5rem;">Chọn ảnh mẫu từ tập dữ liệu thực tế để kiểm tra ngay lập tức:</p>
        """, unsafe_allow_html=True)

        sample_options = {
            "-- Chọn ảnh mẫu thử nghiệm --": None,
            "Táo tươi (FreshApple)": "FreshApple",
            "Chuối tươi (FreshBanana)": "FreshBanana",
            "Cam tươi (FreshOrange)": "FreshOrange",
            "Dâu tây tươi (FreshStrawberry)": "FreshStrawberry",
            "Táo hỏng (RottenApple)": "RottenApple",
            "Chuối hỏng (RottenBanana)": "RottenBanana",
            "Cam hỏng (RottenOrange)": "RottenOrange",
            "Dâu tây hỏng (RottenStrawberry)": "RottenStrawberry"
        }

        selected_sample_key = st.selectbox("Danh sách ảnh mẫu:", list(sample_options.keys()))
        selected_class = sample_options[selected_sample_key]

        if selected_class and ORIG_DATA_DIR.exists():
            class_folder = ORIG_DATA_DIR / selected_class
            if class_folder.exists():
                imgs = list(class_folder.glob("*.*"))
                if imgs:
                    sample_image_path = imgs[0]

        st.markdown("</div>", unsafe_allow_html=True)

    # Xác định ảnh mục tiêu
    target_image = None
    image_source_name = ""

    if uploaded_file is not None:
        try:
            target_image = Image.open(uploaded_file).convert("RGB")
            image_source_name = uploaded_file.name
        except Exception as e:
            st.error(f"Không thể mở file ảnh tải lên: {e}")
    elif sample_image_path is not None:
        try:
            target_image = Image.open(sample_image_path).convert("RGB")
            image_source_name = f"Ảnh mẫu: {sample_image_path.name} ({sample_image_path.parent.name})"
        except Exception as e:
            st.error(f"Không thể mở ảnh mẫu: {e}")

    # XỬ LÝ DỰ ĐOÁN KHI CÓ ẢNH
    if target_image is not None:
        st.markdown("---")
        img_col1, img_col2 = st.columns([1, 1.2])

        with img_col1:
            st.markdown(f"**Ảnh đầu vào:** `{image_source_name}`")
            st.image(target_image, use_container_width=True, caption=f"Kích thước gốc: {target_image.size[0]} x {target_image.size[1]} px")

        # Trích xuất HOG
        img_array = np.array(target_image)
        t_hog_start = time.perf_counter()
        features, hog_image = extract_hog_features(img_array, target_size=(64, 64), visualize=True)
        t_hog_ms = (time.perf_counter() - t_hog_start) * 1000

        with img_col2:
            st.markdown(f"**Trực quan hóa HOG Map:** `1,764 chiều ({t_hog_ms:.1f} ms)`")
            if hog_image is not None:
                # Chuẩn hóa min-max về dải [0, 255] kiểu uint8 để hiển thị sắc nét và tương thích hoàn toàn với Streamlit
                h_min = float(np.min(hog_image))
                h_max = float(np.max(hog_image))
                if h_max > h_min:
                    hog_display = np.clip((hog_image - h_min) / (h_max - h_min) * 255.0, 0, 255).astype(np.uint8)
                else:
                    hog_display = np.zeros(hog_image.shape, dtype=np.uint8)
                st.image(hog_display, use_container_width=True, caption="Biểu diễn phân bố hướng và mật độ gradient của biên cạnh (64x64)")

        if features is None:
            st.error("Không thể trích xuất đặc trưng HOG từ bức ảnh này.")
            return

        # Thực thi dự đoán và đo thời gian thực tế
        feat_vector = features.reshape(1, -1)

        # 1. SVM
        t_svm_start = time.perf_counter()
        svm_idx = svm_model.predict(feat_vector)[0]
        svm_time_ms = (time.perf_counter() - t_svm_start) * 1000
        svm_label = labels[svm_idx]
        svm_proba = svm_model.predict_proba(feat_vector)[0] if hasattr(svm_model, "predict_proba") else None
        svm_info = get_label_info(svm_label)

        # 2. Random Forest
        t_rf_start = time.perf_counter()
        rf_idx = rf_model.predict(feat_vector)[0]
        rf_time_ms = (time.perf_counter() - t_rf_start) * 1000
        rf_label = labels[rf_idx]
        rf_proba = rf_model.predict_proba(feat_vector)[0] if hasattr(rf_model, "predict_proba") else None
        rf_info = get_label_info(rf_label)

        # Lưu lịch sử phiên
        st.session_state.prediction_history.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "image": image_source_name,
            "svm_pred": svm_info["vi"],
            "svm_conf": f"{svm_proba[svm_idx]*100:.1f}%" if svm_proba is not None else "N/A",
            "rf_pred": rf_info["vi"],
            "rf_conf": f"{rf_proba[rf_idx]*100:.1f}%" if rf_proba is not None else "N/A",
            "agreement": "Đồng thuận" if svm_label == rf_label else "Khác biệt"
        })

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Đối chiếu kết quả dự đoán")

        # Hai Card kết quả cạnh nhau
        card_col1, card_col2 = st.columns(2)

        # Card SVM
        with card_col1:
            badge_class = "badge-fresh" if svm_info["is_fresh"] else "badge-rotten"
            st.markdown(f"""
            <div class="model-card-svm">
                <div class="model-tag" style="color:#2563EB;">MÔ HÌNH 1: SUPPORT VECTOR MACHINE (SVM)</div>
                <div class="model-pred-title">{svm_info["vi"]}</div>
                <div style="margin-bottom: 0.8rem;">
                    <span class="{badge_class}">Trạng thái: {svm_info["status"]}</span>
                    <span class="badge-neutral">Mã lớp: {svm_label}</span>
                </div>
            """, unsafe_allow_html=True)

            if svm_proba is not None:
                conf = float(svm_proba[svm_idx]) * 100
                st.write(f"**Độ tự tin (Confidence):** `{conf:.2f}%`")
                st.progress(min(max(conf / 100.0, 0.0), 1.0))

                st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#475569;'>Top 3 nhãn có xác suất cao nhất:</span>", unsafe_allow_html=True)
                top_svm = np.argsort(svm_proba)[::-1][:3]
                for i in top_svm:
                    info = get_label_info(labels[i])
                    st.write(f"- **{info['vi']}** (`{labels[i]}`): `{svm_proba[i]*100:.1f}%`")

            st.caption(f"⚡ Thời gian suy luận: {svm_time_ms:.2f} ms")
            st.markdown("</div>", unsafe_allow_html=True)

        # Card Random Forest
        with card_col2:
            badge_class = "badge-fresh" if rf_info["is_fresh"] else "badge-rotten"
            st.markdown(f"""
            <div class="model-card-rf">
                <div class="model-tag" style="color:#16A34A;">MÔ HÌNH 2: RANDOM FOREST (RF)</div>
                <div class="model-pred-title">{rf_info["vi"]}</div>
                <div style="margin-bottom: 0.8rem;">
                    <span class="{badge_class}">Trạng thái: {rf_info["status"]}</span>
                    <span class="badge-neutral">Mã lớp: {rf_label}</span>
                </div>
            """, unsafe_allow_html=True)

            if rf_proba is not None:
                conf = float(rf_proba[rf_idx]) * 100
                st.write(f"**Độ tự tin (Confidence):** `{conf:.2f}%`")
                st.progress(min(max(conf / 100.0, 0.0), 1.0))

                st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#475569;'>Top 3 nhãn có xác suất cao nhất:</span>", unsafe_allow_html=True)
                top_rf = np.argsort(rf_proba)[::-1][:3]
                for i in top_rf:
                    info = get_label_info(labels[i])
                    st.write(f"- **{info['vi']}** (`{labels[i]}`): `{rf_proba[i]*100:.1f}%`")

            st.caption(f"⚡ Thời gian suy luận: {rf_time_ms:.2f} ms")
            st.markdown("</div>", unsafe_allow_html=True)

        # Banner đối chiếu
        if svm_label == rf_label:
            st.markdown(f"""
            <div class="consensus-agree">
                <b>🤝 Hai mô hình đạt được sự đồng thuận cao:</b><br>
                Cả <b>SVM</b> và <b>Random Forest</b> đều dự đoán bức ảnh này là <b>{svm_info['vi']}</b> ({svm_info['status']}).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="consensus-disagree">
                <b>⚡ Có sự khác biệt dự đoán giữa hai mô hình:</b><br>
                - <b>SVM</b> dự đoán: <b>{svm_info['vi']}</b> ({svm_info['status']})<br>
                - <b>Random Forest</b> dự đoán: <b>{rf_info['vi']}</b> ({rf_info['status']})<br>
                <i>Lưu ý: Cần đối chiếu với nhãn thực tế của ảnh để xác định mô hình nào dự đoán chính xác hơn trong trường hợp này.</i>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Empty State
        st.markdown("""
        <div style="text-align:center; padding: 3.5rem 1rem; background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 16px; margin-top: 1rem;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <h3 style="color:#475569; font-weight:700; margin: 1rem 0 0.4rem 0;">Chưa có hình ảnh được chọn</h3>
            <p style="color:#94A3B8; font-size: 0.92rem; max-width: 480px; margin: 0 auto;">Hãy kéo thả file ảnh trái cây vào khung tải ảnh bên trên hoặc bấm chọn nhanh một ảnh mẫu để hệ thống phân loại tự động.</p>
        </div>
        """, unsafe_allow_html=True)

    # Lịch sử phân loại trong phiên
    if st.session_state.prediction_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🕒 Lịch sử thử nghiệm trong phiên làm việc")
        df_hist = pd.DataFrame(st.session_state.prediction_history[::-1])
        st.dataframe(
            df_hist,
            use_container_width=True,
            column_config={
                "timestamp": "Thời gian",
                "image": "Tên ảnh",
                "svm_pred": "Dự đoán SVM",
                "svm_conf": "Tự tin SVM",
                "rf_pred": "Dự đoán RF",
                "rf_conf": "Tự tin RF",
                "agreement": "Đối chiếu"
            },
            hide_index=True
        )


# ==============================================================================
# TRANG 3: SO SÁNH MÔ HÌNH (SCIENTIFIC EVALUATION)
# ==============================================================================
def render_comparison_page(metrics_data):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 800; color: #1E293B; margin: 0 0 0.4rem 0;">So sánh và phân tích mô hình học máy</h2>
        <p style="color: #64748B; margin: 0; font-size: 0.95rem;">Đánh giá thực nghiệm chi tiết giữa Support Vector Machine và Random Forest trên tập kiểm thử độc lập (Test Set).</p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics_data:
        st.info("Chưa tìm thấy tệp `evaluation_metrics.json`. Hãy chạy `python train_model.py` để tạo báo cáo.")
        return

    svm_m = metrics_data["svm"]
    rf_m = metrics_data["random_forest"]

    # 1. Bảng số liệu tổng hợp
    st.markdown("#### 1. Bảng tổng hợp các chỉ số thực nghiệm trên tập Test (640 mẫu)")

    df_metrics = pd.DataFrame({
        "Chỉ số đánh giá": [
            "Độ chính xác tổng quan (Accuracy)",
            "Độ chuẩn xác trung bình (Precision Macro)",
            "Độ thu hồi trung bình (Recall Macro)",
            "F1-Score trung bình (F1 Macro)",
            "Thời gian huấn luyện (Training Time)",
            "Độ trễ suy luận trung bình (Inference Latency)"
        ],
        "Support Vector Machine (SVM)": [
            f"{svm_m['accuracy']*100:.2f}%",
            f"{svm_m['precision_macro']*100:.2f}%",
            f"{svm_m['recall_macro']*100:.2f}%",
            f"{svm_m['f1_macro']*100:.2f}%",
            f"{svm_m.get('training_time_sec', 28.29):.2f} giây",
            f"{svm_m.get('inference_time_ms', 1.25):.2f} ms"
        ],
        "Random Forest Classifier": [
            f"{rf_m['accuracy']*100:.2f}%",
            f"{rf_m['precision_macro']*100:.2f}%",
            f"{rf_m['recall_macro']*100:.2f}%",
            f"{rf_m['f1_macro']*100:.2f}%",
            f"{rf_m.get('training_time_sec', 1.57):.2f} giây",
            f"{rf_m.get('inference_time_ms', 2.80):.2f} ms"
        ],
        "Chênh lệch (SVM vs RF)": [
            f"+{(svm_m['accuracy'] - rf_m['accuracy'])*100:.2f}% (SVM cao hơn)",
            f"+{(svm_m['precision_macro'] - rf_m['precision_macro'])*100:.2f}% (SVM cao hơn)",
            f"+{(svm_m['recall_macro'] - rf_m['recall_macro'])*100:.2f}% (SVM cao hơn)",
            f"+{(svm_m['f1_macro'] - rf_m['f1_macro'])*100:.2f}% (SVM cao hơn)",
            f"+{svm_m.get('training_time_sec', 28.29) - rf_m.get('training_time_sec', 1.57):.2f}s (RF nhanh hơn)",
            f"-{rf_m.get('inference_time_ms', 2.80) - svm_m.get('inference_time_ms', 1.25):.2f}ms (SVM nhanh hơn)"
        ]
    })
    st.table(df_metrics)

    # 2. Hai biểu đồ so sánh: Chỉ số & Thời gian
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 2. Biểu đồ so sánh 4 chỉ số chất lượng")
        if COMP_CHART_PATH.exists():
            st.image(str(COMP_CHART_PATH), use_container_width=True)
        else:
            st.info("Chưa có biểu đồ so sánh.")

    with col_chart2:
        st.markdown("#### 3. So sánh thời gian huấn luyện (giây)")
        fig_time, ax_time = plt.subplots(figsize=(6, 4.3), facecolor='#FFFFFF')
        models = ["Random Forest", "SVM (RBF)"]
        times = [rf_m.get("training_time_sec", 1.57), svm_m.get("training_time_sec", 28.29)]
        colors = ["#10B981", "#2563EB"]

        bars = ax_time.barh(models, times, color=colors, height=0.45, edgecolor="none")
        ax_time.set_xlabel("Thời gian (giây) - Càng thấp càng tốt", fontsize=9, fontweight="600", color="#475569")
        ax_time.grid(axis="x", linestyle="--", alpha=0.35, color="#94A3B8")
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)
        ax_time.spines['left'].set_color("#CBD5E1")
        ax_time.spines['bottom'].set_color("#CBD5E1")

        for b in bars:
            w = b.get_width()
            ax_time.annotate(f"{w:.2f}s", xy=(w, b.get_y() + b.get_height()/2), xytext=(5, 0),
                             textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="700", color="#1E293B")
        fig_time.tight_layout()
        st.pyplot(fig_time)
        plt.close(fig_time)

    st.markdown("---")

    # 4. Ma trận nhầm lẫn
    st.markdown("#### 4. Ma trận nhầm lẫn (Confusion Matrix) trên 16 lớp")
    tab_svm_cm, tab_rf_cm = st.tabs(["🔹 Ma trận nhầm lẫn SVM", "🌲 Ma trận nhầm lẫn Random Forest"])

    with tab_svm_cm:
        if CM_SVM_PATH.exists():
            st.image(str(CM_SVM_PATH), use_container_width=True, caption="Confusion Matrix - SVM (Kernel RBF)")
        else:
            st.info("Chưa có hình ảnh ma trận nhầm lẫn SVM.")

    with tab_rf_cm:
        if CM_RF_PATH.exists():
            st.image(str(CM_RF_PATH), use_container_width=True, caption="Confusion Matrix - Random Forest Classifier")
        else:
            st.info("Chưa có hình ảnh ma trận nhầm lẫn Random Forest.")

    st.markdown("---")

    # 5. Phân tích học thuật
    st.markdown(r"""
    #### 5. Phân tích học thuật & Giải thích kết quả

    *   **Tại sao SVM lại đạt độ chính xác cao hơn Random Forest (+9.07%)?**
        1.  **Đặc tính của HOG:** Vector đặc trưng HOG có số chiều lớn ($1,764$ chiều), biểu diễn mật độ gradient liên tục của hình dáng quả. SVM với hàm nhân RBF (Radial Basis Function) có khả năng ánh xạ các mẫu vào không gian đặc trưng vô hạn chiều để tìm siêu phẳng phân tách tối ưu với khoảng cách lề cực đại (Max Margin).
        2.  **Cơ chế của Random Forest:** Random Forest phân tách dựa trên các cây quyết định trực giao từng trục (axis-aligned splits) và chỉ lấy một tập con các đặc trưng tại mỗi node ($m = \sqrt{p}$). Khi đặc trưng là các mẫu biên cạnh phụ thuộc lẫn nhau, việc chia cắt từng trục độc lập dễ làm mất đi cấu trúc hình học tổng thể của quả.
    *   **Về chi phí tính toán & Tốc độ huấn luyện:**
        *   **Random Forest** có ưu thế vượt trội về tốc độ huấn luyện ($1.57$ giây so với $28.29$ giây của SVM - nhanh hơn 18 lần) nhờ cơ chế xây dựng các cây độc lập có thể phân tán trên đa nhân CPU (`n_jobs=-1`).
        *   **SVM** cần giải bài toán quy hoạch toàn phương $O(N^2 \cdot D)$ và thực hiện kiểm định chéo Platt scaling để tính xác suất, khiến thời gian huấn luyện lâu hơn.
    """)


# ==============================================================================
# TRANG 4: DỮ LIỆU VÀ ĐÁNH GIÁ (DATASET & METRICS)
# ==============================================================================
def render_dataset_page(metrics_data, labels):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 800; color: #1E293B; margin: 0 0 0.4rem 0;">Dữ liệu và quy trình trích xuất đặc trưng</h2>
        <p style="color: #64748B; margin: 0; font-size: 0.95rem;">Chi tiết về cấu trúc dataset thực nghiệm, phương pháp tiền xử lý và cơ chế trích xuất HOG.</p>
    </div>
    """, unsafe_allow_html=True)

    # Thống kê dataset
    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Bộ dữ liệu thực tế</div>
            <div class="kpi-value">3,200</div>
            <div class="kpi-subtext">Ảnh gốc (Original Image)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Phân bố Train / Test</div>
            <div class="kpi-value">80% - 20%</div>
            <div class="kpi-subtext">2,560 Train / 640 Test (Stratified)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_d3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Kích thước vector HOG</div>
            <div class="kpi-value">1,764</div>
            <div class="kpi-subtext">Đặc trưng hình học / ảnh 64x64</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bảng kê 16 lớp
    st.markdown("#### 1. Danh mục 16 lớp trái cây trong hệ thống")

    class_list = labels if labels else list(CLASS_METADATA.keys())
    table_rows = []
    for lbl in sorted(class_list):
        info = get_label_info(lbl)
        table_rows.append({
            "Mã lớp (Class ID)": lbl,
            "Tên tiếng Việt": info["vi"],
            "Loại quả": info["fruit"],
            "Tình trạng chất lượng": info["status"],
            "Số mẫu ảnh gốc": 200,
            "Tập Train (80%)": 160,
            "Tập Test (20%)": 40
        })

    df_classes = pd.DataFrame(table_rows)
    st.dataframe(df_classes, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Quy trình 3 bước
    st.markdown("#### 2. Quy trình xử lý và huấn luyện chi tiết")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1B4332; margin-bottom:0.6rem;">BƯỚC 1: TIỀN XỬ LÝ ẢNH</div>
            <ul style="font-size:0.88rem; color:#475569; padding-left:1.2rem; line-height:1.6;">
                <li>Đọc ảnh an toàn kháng lỗi Windows path bằng <code>PIL</code>.</li>
                <li>Sử dụng kỹ thuật nén giải mã <code>PIL draft</code> tăng tốc gấp 7 lần.</li>
                <li>Resize chuẩn hóa về kích thước <b>64 x 64 pixel</b>.</li>
                <li>Chuyển đổi sang ảnh mức xám (Grayscale).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1B4332; margin-bottom:0.6rem;">BƯỚC 2: TRÍCH XUẤT HOG</div>
            <ul style="font-size:0.88rem; color:#475569; padding-left:1.2rem; line-height:1.6;">
                <li><b>Cell size:</b> 8 x 8 px (ảnh có 8x8 = 64 cells).</li>
                <li><b>Block size:</b> 2 x 2 cells (chuẩn hóa lân cận).</li>
                <li><b>Orientations:</b> 9 bins (từ 0° đến 180°).</li>
                <li><b>Block normalization:</b> L2-Hys.</li>
                <li><b>Tổng số chiều:</b> 7 × 7 × 4 × 9 = <b>1,764 features</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1B4332; margin-bottom:0.6rem;">BƯỚC 3: HUẤN LUYỆN ML</div>
            <ul style="font-size:0.88rem; color:#475569; padding-left:1.2rem; line-height:1.6;">
                <li>Phân chia tập <b>Train 80% / Test 20%</b> phân tầng (Stratified).</li>
                <li><b>SVM:</b> RBF Kernel, C=10.0, tính xác suất Platt scaling.</li>
                <li><b>Random Forest:</b> 100 cây quyết định, tối đa hóa thông tin Gini.</li>
                <li>Lưu trữ mô hình dạng <code>.joblib</code> vào <code>models/</code>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TRANG 5: GIỚI THIỆU ĐỀ TÀI (PROJECT OVERVIEW & PIPELINE)
# ==============================================================================
def render_about_page():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 800; color: #1E293B; margin: 0 0 0.4rem 0;">Giới thiệu đề tài & Kiến trúc hệ thống</h2>
        <p style="color: #64748B; margin: 0; font-size: 0.95rem;">Báo cáo học thuật đề tài môn Học máy (Machine Learning): Phân loại trái cây từ hình ảnh bằng SVM và Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sơ đồ pipeline
    st.markdown("#### 1. Sơ đồ kiến trúc xử lý tuần tự (End-to-End ML Pipeline)")
    st.markdown("""
    ```text
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                               ẢNH ĐẦU VÀO                                   │
    │          (Ảnh chụp quả tươi hoặc quả hỏng: JPG, PNG, WEBP)                  │
    └──────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        GIAI ĐOẠN 1: TIỀN XỬ LÝ ẢNH                          │
    │  - Giải mã ảnh an toàn bằng PIL (Khắc phục lỗi Windows path)                 │
    │  - Chuẩn hóa kích thước: Resize 64 x 64 pixels                               │
    │  - Chuyển đổi không gian màu RGB sang Grayscale                              │
    └──────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │               GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG BIÊN HOG                    │
    │  - Tính toán đạo hàm Gradient theo trục X và Y                              │
    │  - Chia ảnh thành các ô (Cell: 8x8) và nhóm ô (Block: 2x2)                  │
    │  - Lập biểu đồ tần suất góc hướng Gradient (9 orientations)                 │
    │  - Chuẩn hóa L2-Hys tạo ra vector đặc trưng 1,764 chiều                     │
    └───────────────────┬─────────────────────────────────────┬───────────────────┘
                        │                                     │
                        ▼                                     ▼
    ┌───────────────────────────────────────┐ ┌───────────────────────────────────┐
    │     MÔ HÌNH 1: SUPPORT VECTOR MACHINE │ │ MÔ HÌNH 2: RANDOM FOREST          │
    │  - Kernel: RBF (Radial Basis Function)│ │ - Số cây quyết định: 100 cây      │
    │  - Tham số điều hòa: C = 10.0         │ │ - Tiêu chuẩn phân nhánh: Gini     │
    │  - Dự đoán nhãn & Tính xác suất       │ │ - Tối ưu đa luồng CPU: n_jobs=-1  │
    └───────────────────┬───────────────────┘ └───────────────────┬───────────────┘
                        │                                         │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                  GIAI ĐOẠN 3: ĐỐI CHIẾU & ĐÁNH GIÁ KẾT QUẢ                  │
    │  - So sánh nhãn dự đoán giữa hai mô hình (Đồng thuận / Khác biệt)           │
    │  - Đo thời gian suy luận (Latency) tính bằng mili-giây                      │
    │  - Đánh giá tổng thể qua Accuracy, Precision, Recall và F1-Score            │
    └─────────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Cơ sở lý thuyết
    st.markdown("#### 2. Cơ sở lý thuyết của các phương pháp sử dụng")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1E293B; margin-bottom:0.6rem;">Đặc trưng HOG</div>
            <p style="font-size:0.88rem; color:#475569; line-height:1.55;">
                <b>Histogram of Oriented Gradients (HOG)</b> là phương pháp trích xuất đặc trưng kinh điển trong thị giác máy tính. HOG nắm bắt hình dáng và đường biên vật thể bằng cách đếm số lần xuất hiện của các hướng gradient trong các vùng cục bộ, có khả năng bất biến với thay đổi ánh sáng và bóng mờ.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown(r"""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1E293B; margin-bottom:0.6rem;">Support Vector Machine</div>
            <p style="font-size:0.88rem; color:#475569; line-height:1.55;">
                <b>SVM</b> tìm kiếm siêu phẳng có khoảng cách cực đại (Maximum Margin) giữa các lớp. Với hàm nhân RBF ($K(x, x') = \exp(-\gamma ||x - x'||^2)$), mô hình có khả năng phân tách phi tuyến cực mạnh trên không gian đặc trưng nhiều chiều của HOG.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="dashboard-card" style="height:100%;">
            <div style="font-weight:700; color:#1E293B; margin-bottom:0.6rem;">Random Forest</div>
            <p style="font-size:0.88rem; color:#475569; line-height:1.55;">
                <b>Random Forest</b> là thuật toán học kết hợp (Ensemble Learning) hoạt động dựa trên phương pháp Bagging. Mô hình xây dựng 100 cây quyết định độc lập trên các tập con mẫu và đặc trưng ngẫu nhiên, sau đó tổng hợp kết quả bằng cơ chế bỏ phiếu đa số.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Câu hỏi vấn đáp thường gặp (FAQ)
    st.markdown("#### 3. Câu hỏi thảo luận và vấn đáp thường gặp")
    with st.expander("❓ Tại sao không dùng trực tiếp giá trị pixel ảnh mà phải qua HOG?"):
        st.write("""
        Giá trị pixel gốc (Raw RGB) phụ thuộc quá nhiều vào độ sáng, góc chụp và màu nền. Nếu dùng pixel phẳng, vector đầu vào rất dễ bị nhiễu và dễ overfit. Đặc trưng HOG chỉ tập trung vào gradient (sự biến thiên biên cạnh hình học), giúp mô hình nhận diện được quả tròn, quả dài, vết đốm sần sùi của quả thối một cách khái quát hóa hơn.
        """)

    with st.expander("❓ Tại sao SVM huấn luyện lâu hơn Random Forest?"):
        st.write("""
        SVM giải bài toán tối ưu hóa lồi toàn phương (Quadratic Programming) với độ phức tạp tính toán xấp xỉ $O(N^2)$ đến $O(N^3)$ theo số lượng mẫu. Thêm vào đó, để xuất ra xác suất confidence, hệ thống chạy phương pháp Platt Scaling với kiểm định chéo nội bộ. Trong khi đó, Random Forest xây dựng 100 cây quyết định độc lập song song trên đa luồng CPU nên tốc độ huấn luyện nhanh gấp 18 lần.
        """)

    with st.expander("❓ Làm thế nào để cải thiện độ chính xác hơn nữa trong tương lai?"):
        st.write(r"""
        1. Kết hợp đặc trưng màu sắc (Color Histogram trong không gian HSV) với HOG vì màu sắc quả tươi và quả thối có sự khác biệt rõ nét.
        2. Tinh chỉnh siêu tham số (Hyperparameter Tuning bằng GridSearchCV) cho $C$ và $\gamma$ của SVM.
        3. Mở rộng thử nghiệm với mạng nơ-ron tích chập (CNN - Convolutional Neural Networks) như MobileNetV2 để tự động học đặc trưng.
        """)


# ==============================================================================
# HÀM CHÍNH (MAIN ENTRYPOINT)
# ==============================================================================
def main():
    svm_model, rf_model, labels = load_ml_models()
    metrics_data = load_metrics_data()

    # Render Sidebar
    render_sidebar(svm_model, rf_model, labels, metrics_data)

    # Điều hướng trang
    current_page = st.session_state.current_page

    if current_page == "📌 Tổng quan":
        render_overview_page(metrics_data, labels)
    elif current_page == "🔍 Phân loại trái cây":
        render_classification_page(svm_model, rf_model, labels)
    elif current_page == "📊 So sánh mô hình":
        render_comparison_page(metrics_data)
    elif current_page == "📁 Dữ liệu và đánh giá":
        render_dataset_page(metrics_data, labels)
    elif current_page == "📖 Giới thiệu đề tài":
        render_about_page()


if __name__ == "__main__":
    main()
