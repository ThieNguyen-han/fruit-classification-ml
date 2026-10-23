"""
train_model.py - Script đọc dữ liệu, trích xuất đặc trưng HOG,
huấn luyện hai mô hình SVM và Random Forest, đánh giá và so sánh kết quả.

Đề tài: Phân loại trái cây từ hình ảnh bằng SVM và Random Forest để so sánh độ chính xác.
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# Import module trích xuất đặc trưng từ src
from src.features import load_dataset_features


# Định nghĩa các đường dẫn thư mục chuẩn
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def ensure_directories():
    """Đảm bảo các thư mục cần thiết đều tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_and_save_confusion_matrix(cm, class_names, model_name, save_path):
    """Vẽ và lưu biểu đồ ma trận nhầm lẫn (Confusion Matrix)."""
    fig_size = max(10, len(class_names) * 0.75)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8 if len(class_names) > 10 else 11}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=12)
    plt.xlabel("Dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Thực tế (Actual)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_and_save_comparison_chart(svm_metrics, rf_metrics, save_path):
    """Vẽ biểu đồ cột so sánh các chỉ số giữa SVM và Random Forest."""
    metrics_keys = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    svm_values = [
        svm_metrics["accuracy"],
        svm_metrics["precision_macro"],
        svm_metrics["recall_macro"],
        svm_metrics["f1_macro"]
    ]
    rf_values = [
        rf_metrics["accuracy"],
        rf_metrics["precision_macro"],
        rf_metrics["recall_macro"],
        rf_metrics["f1_macro"]
    ]

    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, svm_values, width, label="SVM (RBF Kernel)", color="#4A90E2")
    rects2 = ax.bar(x + width/2, rf_values, width, label="Random Forest", color="#50E3C2")

    ax.set_ylabel("Điểm số (0.0 - 1.0)", fontsize=12)
    ax.set_title("So sánh hiệu năng giữa SVM và Random Forest", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print("=" * 70)
    print(" HỆ THỐNG PHÂN LOẠI TRÁI CÂY TỪ HÌNH ẢNH (SVM & RANDOM FOREST)")
    print("=" * 70)

    ensure_directories()

    # 1. Nạp và trích xuất đặc trưng từ thư mục data/
    X, y, label_names = load_dataset_features(data_dir=DATA_DIR, target_size=(64, 64))

    # 2. Kiểm tra điều kiện dữ liệu (Xử lý an toàn nếu chưa có dataset)
    if X.size == 0 or len(label_names) < 2:
        print("\n" + "!" * 70)
        print(" [HƯỚNG DẪN] CHƯA CÓ ĐỦ DỮ LIỆU ĐỂ HUẤN LUYỆN MÔ HÌNH")
        print("!" * 70)
        print("Hệ thống yêu cầu ít nhất 2 thư mục lớp trái cây trong thư mục 'data/'")
        print("Ví dụ cấu trúc thư mục chuẩn:")
        print("  data/")
        print("    ├── apple/        (Chứa các file ảnh quả táo: apple1.jpg, ...)")
        print("    ├── banana/       (Chứa các file ảnh quả chuối: banana1.jpg, ...)")
        print("    └── orange/       (Chứa các file ảnh quả cam: orange1.jpg, ...)")
        print("\nCách thực hiện:")
        print("1. Tải dataset ảnh trái cây (ví dụ: Fruit-360 từ Kaggle).")
        print("2. Giải nén hoặc sao chép các thư mục ảnh vào thư mục 'data/'.")
        print("3. Chạy lại lệnh: python train_model.py")
        print("!" * 70 + "\n")
        sys.exit(0)

    # Kiểm tra số lượng mẫu tối thiểu cho mỗi lớp
    unique_labels, counts = np.unique(y, return_counts=True)
    if np.min(counts) < 5:
        print("\n[Cảnh báo] Mỗi lớp trái cây cần có ít nhất 5 ảnh để chia tập train/test hợp lệ.")
        print(f"Chi tiết số lượng hiện có: {dict(zip(label_names, counts))}")
        sys.exit(0)

    print(f"\nPhân bố dữ liệu: {len(X)} mẫu trên {len(label_names)} lớp trái cây.")
    for name, cnt in zip(label_names, counts):
        print(f"  - {name}: {cnt} ảnh")

    # 3. Phân chia tập huấn luyện và kiểm thử (Train/Test Split 80/20)
    print("\nChia dữ liệu Train/Test tỉ lệ 80% - 20% (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Tập Train: {X_train.shape[0]} mẫu")
    print(f"  - Tập Test:  {X_test.shape[0]} mẫu")

    # 4. Huấn luyện mô hình SVM (Support Vector Machine)
    print("\n" + "-" * 50)
    print("1/2 Đang huấn luyện Support Vector Machine (SVM - Kernel RBF)...")
    svm_clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)

    # 5. Huấn luyện mô hình Random Forest
    print("2/2 Đang huấn luyện Random Forest Classifier (100 cây quyết định)...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("-" * 50)

    # 6. Đánh giá chi tiết các chỉ số (Accuracy, Precision, Recall, F1-score)
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    svm_metrics = compute_metrics(y_test, y_pred_svm)
    rf_metrics = compute_metrics(y_test, y_pred_rf)

    # In bảng so sánh trực quan ra Terminal
    print("\n" + "=" * 65)
    print(f"{'Chỉ số đánh giá':<25} | {'SVM':<16} | {'Random Forest':<16}")
    print("=" * 65)
    print(f"{'Accuracy':<25} | {svm_metrics['accuracy'] * 100:>14.2f}% | {rf_metrics['accuracy'] * 100:>14.2f}%")
    print(f"{'Precision (Macro)':<25} | {svm_metrics['precision_macro'] * 100:>14.2f}% | {rf_metrics['precision_macro'] * 100:>14.2f}%")
    print(f"{'Recall (Macro)':<25} | {svm_metrics['recall_macro'] * 100:>14.2f}% | {rf_metrics['recall_macro'] * 100:>14.2f}%")
    print(f"{'F1-Score (Macro)':<25} | {svm_metrics['f1_macro'] * 100:>14.2f}% | {rf_metrics['f1_macro'] * 100:>14.2f}%")
    print(f"{'F1-Score (Weighted)':<25} | {svm_metrics['f1_weighted'] * 100:>14.2f}% | {rf_metrics['f1_weighted'] * 100:>14.2f}%")
    print("=" * 65)

    # 7. Lưu các mô hình và nhãn vào thư mục models/
    svm_path = MODELS_DIR / "svm_model.joblib"
    rf_path = MODELS_DIR / "rf_model.joblib"
    labels_path = MODELS_DIR / "labels.joblib"

    print(f"\nLưu các mô hình đã huấn luyện:")
    joblib.dump(svm_clf, svm_path)
    print(f" -> Đã lưu SVM: {svm_path.name}")
    joblib.dump(rf_clf, rf_path)
    print(f" -> Đã lưu Random Forest: {rf_path.name}")
    joblib.dump(label_names, labels_path)
    print(f" -> Đã lưu danh sách nhãn: {labels_path.name}")

    # 8. Lưu kết quả đánh giá và biểu đồ vào results/
    results_summary = {
        "classes": label_names,
        "dataset_summary": {
            "total_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0])
        },
        "svm": svm_metrics,
        "random_forest": rf_metrics
    }

    metrics_json_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    print(f" -> Đã lưu số liệu metrics: {metrics_json_path.name}")

    # Tạo confusion matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_and_save_confusion_matrix(cm_svm, label_names, "SVM Classifier", RESULTS_DIR / "confusion_matrix_svm.png")
    plot_and_save_confusion_matrix(cm_rf, label_names, "Random Forest Classifier", RESULTS_DIR / "confusion_matrix_rf.png")
    print(" -> Đã xuất biểu đồ ma trận nhầm lẫn (confusion_matrix_*.png)")

    # Tạo biểu đồ so sánh
    plot_and_save_comparison_chart(svm_metrics, rf_metrics, RESULTS_DIR / "model_comparison.png")
    print(" -> Đã xuất biểu đồ so sánh (model_comparison.png)")

    # Báo cáo chi tiết dạng text
    report_text_path = RESULTS_DIR / "classification_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION REPORT: SVM ===\n")
        f.write(classification_report(y_test, y_pred_svm, target_names=label_names, zero_division=0))
        f.write("\n\n=== CLASSIFICATION REPORT: RANDOM FOREST ===\n")
        f.write(classification_report(y_test, y_pred_rf, target_names=label_names, zero_division=0))
    print(f" -> Đã lưu báo cáo chi tiết: {report_text_path.name}")

    print("\n[HOÀN THÀNH] Quá trình huấn luyện và đánh giá đã kết thúc thành công!")
    print("Bạn có thể khởi động ứng dụng web demo bằng lệnh: streamlit run app.py")


if __name__ == "__main__":
    main()
