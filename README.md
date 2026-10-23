# Đề Tài: Xây dựng hệ thống phân loại trái cây từ hình ảnh bằng SVM và Random Forest để so sánh độ chính xác

Bài tập lớn môn **Học máy (Machine Learning)**.

---

## 📌 Giới thiệu đề tài
Dự án tập trung vào việc tiền xử lý hình ảnh, trích xuất đặc trưng dạng hình dáng/đường biên sử dụng phương pháp **HOG (Histogram of Oriented Gradients)**, sau đó tiến hành huấn luyện và so sánh hiệu năng giữa hai thuật toán Machine Learning kinh điển:
- **Support Vector Machine (SVM)** với RBF Kernel.
- **Random Forest Classifier (Rừng ngẫu nhiên)** gồm tập hợp nhiều cây quyết định.

Dự án được thiết kế tối ưu cho máy tính không có card đồ họa rời (CPU-only), bộ nhớ RAM 8GB, chạy mượt mà trên hệ điều hành Windows và Python 3.11+.

---

## 📁 Cấu trúc thư mục dự án

```text
fruit-classification-ml/
├── data/                       # Thư mục chứa dữ liệu ảnh (mỗi loại quả là một thư mục con)
│   ├── README.md               # Hướng dẫn chi tiết cách tổ chức dataset
│   └── .gitkeep
├── models/                     # Thư mục lưu trữ mô hình sau khi huấn luyện (.joblib)
│   └── .gitkeep
├── results/                    # Thư mục lưu kết quả đánh giá (JSON, ma trận nhầm lẫn, biểu đồ so sánh)
│   └── .gitkeep
├── src/                        # Mã nguồn các module xử lý nền tảng
│   ├── __init__.py
│   └── features.py             # Tiền xử lý ảnh và trích xuất đặc trưng HOG
├── app.py                      # Ứng dụng web tương tác bằng Streamlit
├── train_model.py              # Script nạp dữ liệu, huấn luyện SVM & RF, đánh giá và lưu kết quả
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── .gitignore                  # Cấu hình loại trừ file rác, dataset và mô hình nặng
└── README.md                   # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng từng bước (Cho người mới trên Windows & Antigravity IDE)

### Bước 1: Mở Terminal trong Antigravity IDE
Trong Antigravity IDE, bạn có thể mở Terminal bằng phím tắt ``Ctrl + ` `` (hoặc vào menu **Terminal** -> **New Terminal**).

### Bước 2: Tạo và kích hoạt môi trường ảo Python (Khuyên dùng)
Để các thư viện của dự án không bị xung đột với hệ thống, hãy tạo một môi trường ảo `venv`:

```powershell
# Tạo môi trường ảo venv với Python 3.11
python -m venv venv

# Kích hoạt môi trường ảo trên Windows PowerShell
.\venv\Scripts\Activate.ps1
```
*(Nếu gặp lỗi script execution policy trên PowerShell, bạn có thể chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi chạy lại lệnh kích hoạt).*

### Bước 3: Cài đặt các thư viện cần thiết
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖼️ Hướng dẫn chuẩn bị dữ liệu (Dataset)

Hiện tại dự án chưa chứa ảnh để đảm bảo dung lượng nhẹ. Khi bắt đầu huấn luyện, bạn hãy tải dataset về và đặt vào thư mục `data/` theo định dạng sau:

```text
data/
├── apple/
│   ├── img01.jpg
│   ├── img02.jpg
│   └── ...
├── banana/
│   ├── img01.jpg
│   └── ...
└── orange/
    ├── img01.jpg
    └── ...
```

> **Gợi ý nguồn dataset miễn phí chất lượng cao:**
> - [Fruits 360 Dataset trên Kaggle](https://www.kaggle.com/datasets/moltean/fruits) (Chỉ cần lấy 3 - 8 loại quả tiêu biểu để huấn luyện nhanh).
> - Hoặc chụp/tự thu thập các ảnh quả rõ nét với định dạng `.jpg`, `.png`.

---

## 🏋️‍♂️ Huấn luyện mô hình và so sánh độ chính xác

Sau khi đã sao chép ảnh vào `data/`, chạy lệnh sau để bắt đầu trích xuất HOG và huấn luyện:

```powershell
python train_model.py
```

### Các bước script tự động thực hiện:
1. Đọc và chuẩn hóa ảnh về kích thước `64x64`, chuyển mức xám.
2. Trích xuất vector đặc trưng HOG (1764 chiều).
3. Chia tập dữ liệu **Train (80%)** và **Test (20%)** phân tầng (stratified).
4. Huấn luyện mô hình **SVM (RBF Kernel)** và **Random Forest**.
5. Tính toán các chỉ số:
   - **Accuracy** (Độ chính xác tổng quan).
   - **Precision (Macro & Weighted)** (Độ chuẩn xác).
   - **Recall (Macro & Weighted)** (Độ thu hồi).
   - **F1-Score (Macro & Weighted)** (Trung bình điều hòa F1).
6. Tự động lưu:
   - File mô hình: `models/svm_model.joblib`, `models/rf_model.joblib`, `models/labels.joblib`.
   - Kết quả chỉ số dạng JSON: `results/evaluation_metrics.json`.
   - Biểu đồ ma trận nhầm lẫn (Confusion Matrix): `results/confusion_matrix_svm.png`, `results/confusion_matrix_rf.png`.
   - Biểu đồ cột so sánh trực quan: `results/model_comparison.png`.
   - Báo cáo chi tiết: `results/classification_report.txt`.

---

## 🌐 Trải nghiệm ứng dụng Web Demo (Streamlit)

Chạy lệnh sau để khởi động giao diện người dùng:

```powershell
streamlit run app.py
```

- Trình duyệt web sẽ tự động mở tại địa chỉ `http://localhost:8501`.
- Giao diện cho phép bạn tải lên một bức ảnh bất kỳ từ máy tính.
- Hệ thống sẽ:
  - Hiển thị ảnh gốc và ảnh trực quan hóa vector biên HOG.
  - Dự đoán đồng thời bằng cả **SVM** và **Random Forest**.
  - Hiển thị xác suất phân bố (Confidence score) của từng mô hình.
  - So sánh xem hai mô hình có đồng thuận kết quả hay không.
- Nếu chưa huấn luyện mô hình, ứng dụng sẽ hiển thị màn hình hướng dẫn từng bước rõ ràng mà không bị lỗi.

---

## 🛡️ Điểm nổi bật kỹ thuật của dự án
1. **Kháng lỗi Windows Path:** Sử dụng `PIL` và `pathlib` xử lý trơn tru đường dẫn có dấu tiếng Việt và dấu cách trên Windows.
2. **Tiết kiệm tài nguyên:** Kích thước HOG vector tối ưu (`np.float32`), nén dữ liệu giúp máy RAM 8GB không bị tràn bộ nhớ.
3. **Xử lý ngoại lệ thân thiện:** Kiểm tra điều kiện dữ liệu trước khi train, thông báo hướng dẫn cụ thể thay vì ngắt đột ngột bằng Traceback.
4. **Minh bạch số liệu:** Không tạo số liệu giả lập, toàn bộ số liệu đánh giá được tính toán trực tiếp từ dữ liệu kiểm thử thực tế.
