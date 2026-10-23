# BÁO CÁO TỔNG KẾT QUÁ TRÌNH THỰC HIỆN DỰ ÁN
## ĐỀ TÀI: HỆ THỐNG PHÂN LOẠI TRÁI CÂY THÔNG MINH (FRUITVISION AI)
### SO SÁNH HIỆU NĂNG GIỮA SUPPORT VECTOR MACHINE (SVM) VÀ RANDOM FOREST TRÊN ĐẶC TRƯNG HOG

---

## 📌 1. TỔNG QUAN DỰ ÁN

*   **Tên hệ thống:** FRUITVISION AI — Hệ thống phân loại trái cây thông minh.
*   **Học phần:** Học máy (Machine Learning).
*   **Mục tiêu nghiên cứu:** 
    1.  Xây dựng quy trình xử lý hình ảnh, trích xuất đặc trưng hình học và đường biên dạng **Histogram of Oriented Gradients (HOG)**.
    2.  Huấn luyện và đánh giá thực nghiệm hai thuật toán học máy kinh điển: **Support Vector Machine (SVM)** với hàm nhân RBF và **Random Forest Classifier (Rừng ngẫu nhiên)**.
    3.  Phân loại đa lớp đồng thời nhận diện **8 loại trái cây** và kiểm tra trạng thái **Tươi ngon (Fresh)** hay **Hư hỏng (Rotten)** trên tập dữ liệu gồm **16 lớp**.
    4.  Xây dựng ứng dụng web tương tác hoàn chỉnh theo phong cách **AI Dashboard & Scientific Data Visualization** bằng Streamlit.

---

## 📁 2. XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU (DATASET)

### 2.1. Cấu trúc tập dữ liệu
Hệ thống sử dụng bộ dữ liệu phân loại chất lượng trái cây thực tế, sau khi được kiểm tra và xử lý tái cấu trúc tự động:

*   **Tập ảnh gốc (Original Image):** Tổng cộng **3,200 ảnh**, được cân bằng tuyệt đối với **200 ảnh/lớp** cho 16 lớp.
*   **Tập ảnh tăng cường (Augmented Image):** Gồm **12,335 ảnh** được tổ chức riêng biệt trong `data/Augmented Image/`, đảm bảo nguyên tắc khoa học **chống rò rỉ dữ liệu (Anti Data Leakage)**: Tập kiểm thử (Test Set) chỉ sử dụng $100\%$ ảnh thật từ tập ảnh gốc.

### 2.2. Danh mục 16 lớp phân loại
| STT | Mã lớp (Class ID) | Tên tiếng Việt | Tình trạng | Số lượng ảnh gốc | Số mẫu Train (80%) | Số mẫu Test (20%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `FreshApple` | Táo tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 2 | `FreshBanana` | Chuối tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 3 | `FreshGrape` | Nho tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 4 | `FreshGuava` | Ổi tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 5 | `FreshJujube` | Táo ta (Táo tàu) tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 6 | `FreshOrange` | Cam tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 7 | `FreshPomegranate` | Lựu tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 8 | `FreshStrawberry` | Dâu tây tươi | 🟢 Tươi ngon | 200 | 160 | 40 |
| 9 | `RottenApple` | Táo hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 10 | `RottenBanana` | Chuối hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 11 | `RottenGrape` | Nho hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 12 | `RottenGuava` | Ổi hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 13 | `RottenJujube` | Táo ta hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 14 | `RottenOrange` | Cam hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 15 | `RottenPomegranate` | Lựu hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| 16 | `RottenStrawberry` | Dâu tây hỏng | 🔴 Hư hỏng | 200 | 160 | 40 |
| **Tổng** | **16 Lớp** | **8 Loại quả** | **2 Trạng thái** | **3,200** | **2,560** | **640** |

---

## ⚙️ 3. TỐI ƯU HÓA KỸ THUẬT & TRÍCH XUẤT ĐẶC TRƯNG (`src/features.py`)

1.  **Kháng lỗi đường dẫn hệ thống trên Windows:**
    *   Sử dụng thư viện `PIL.Image` và `pathlib.Path` đọc dữ liệu theo luồng nhị phân an toàn, khắc phục triệt để lỗi đường dẫn chứa tiếng Việt có dấu hoặc khoảng trắng mà `cv2.imread` thường gặp.
2.  **Tăng tốc giải mã ảnh gấp 7 lần (`PIL draft`):**
    *   Ảnh gốc có độ phân giải lớn (13 Megapixels: $4160 \times 3120$). Áp dụng phương thức `img.draft('L', (64, 64))` cho phép thư viện JPEG giải nén trực tiếp ở mức DCT thu nhỏ, giảm thời gian nạp từ $110\text{ ms}$ xuống chỉ còn **$15\text{ ms/ảnh}$**.
3.  **Xử lý đa luồng song song (`ThreadPoolExecutor`):**
    *   Tận dụng đa nhân CPU, trích xuất đặc trưng đồng thời cho 3,200 bức ảnh hoàn tất chỉ trong **~13 giây** (thay vì mất nhiều phút khi chạy đơn luồng).
4.  **Cấu hình tham số HOG chuẩn hóa:**
    *   *Kích thước chuẩn hóa:* $64 \times 64$ pixels (ảnh mức xám Grayscale).
    *   *Cell size:* $8 \times 8$ pixels (tổng cộng $8 \times 8 = 64$ cells).
    *   *Block size:* $2 \times 2$ cells (chuẩn hóa lân cận).
    *   *Orientations:* 9 bins ($0^\circ$ đến $180^\circ$).
    *   *Chuẩn hóa khối:* L2-Hys (L2-norm with clipping).
    *   *Số chiều vector đặc trưng:* $(8 - 1) \times (8 - 1) \times (2 \times 2) \times 9 = 7 \times 7 \times 4 \times 9 =$ **1,764 chiều/ảnh**.

---

## 🔬 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ THỰC NGHIỆM (`train_model.py`)

### 4.1. Phân chia tập dữ liệu
*   Áp dụng phương pháp phân tầng **Stratified Split** với tỷ lệ **80% Train / 20% Test** (Random State = 42).
*   Tập huấn luyện (Train): **2,560 mẫu** (160 mẫu/lớp).
*   Tập kiểm thử độc lập (Test): **640 mẫu** (40 mẫu/lớp).

### 4.2. Bảng số liệu so sánh hiệu năng thực tế
Toàn bộ số liệu dưới đây được trích xuất trực tiếp từ tệp kết quả thực nghiệm `results/evaluation_metrics.json`:

| Tiêu chí đánh giá | Support Vector Machine (SVM) | Random Forest Classifier | Chênh lệch (SVM vs RF) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác tổng quan (Accuracy)** | **71.41%** | 62.34% | **+9.07%** (SVM vượt trội) |
| **Độ chuẩn xác (Precision Macro)** | **72.61%** | 62.55% | **+10.06%** (SVM vượt trội) |
| **Độ thu hồi (Recall Macro)** | **71.41%** | 62.34% | **+9.07%** (SVM vượt trội) |
| **F1-Score trung bình (F1 Macro)** | **71.49%** | 61.47% | **+10.02%** (SVM vượt trội) |
| **Thời gian huấn luyện (Training Time)** | 28.29 giây | **1.57 giây** | RF nhanh hơn **18 lần** |
| **Độ trễ suy luận (Inference Latency)** | **1.25 ms/ảnh** | 2.80 ms/ảnh | SVM nhanh hơn gấp đôi |
| **Cấu hình siêu tham số** | Kernel RBF, $C=10.0$, $\gamma=\text{'scale'}$ | 100 Trees, Criterion='gini' | — |

### 4.3. Phân tích học thuật chuyên sâu
*   **Vì sao SVM đạt độ chính xác cao hơn Random Forest (+9.07%)?**
    1.  *Không gian đặc trưng nhiều chiều:* Vector HOG có $1,764$ chiều liên tục thể hiện mật độ gradient biên cạnh. SVM với hàm nhân RBF (Radial Basis Function) ánh xạ hiệu quả dữ liệu vào không gian vô hạn chiều, tìm siêu phẳng phân tách tối ưu có khoảng cách lề lớn nhất (Max Margin).
    2.  *Hạn chế của Random Forest trên HOG:* Cây quyết định trong Random Forest chia cắt không gian theo từng trục độc lập ($m = \sqrt{1764} = 42$ đặc trưng tại mỗi node). Cấu trúc hình học của quả vốn là sự liên kết chặt chẽ giữa các gradient cạnh nhau; việc lấy mẫu ngẫu nhiên từng đặc trưng rời rạc dễ làm phân mảnh cấu trúc toàn cục của quả.
*   **Về chi phí tính toán & Tốc độ huấn luyện:**
    *   Random Forest có khả năng phân tán song song trên toàn bộ nhân CPU (`n_jobs=-1`), hoàn thành huấn luyện 100 cây chỉ trong **1.57 giây**.
    *   SVM giải bài toán quy hoạch toàn phương $O(N^2 \cdot D)$ và thực hiện kiểm định chéo Platt Scaling để tính xác suất, do đó thời gian huấn luyện là **28.29 giây**.
    *   Khi suy luận (Inference), cả hai mô hình đều phản hồi tức thì dưới **3 mili-giây**, đáp ứng hoàn hảo yêu cầu thời gian thực.

---

## 🎨 5. THIẾT KẾ & TRIỂN KHAI GIAO DIỆN WEB FRUITVISION AI (`app.py`)

Ứng dụng được xây dựng theo tiêu chuẩn **AI Dashboard kết hợp Scientific Data Visualization** gồm 5 phân hệ chuyên sâu:

```text
FRUITVISION AI
├── 📌 Thanh điều hướng (Sidebar)
│   ├── Logo Vector SVG FRUITVISION AI & Tagline hệ thống
│   ├── Menu điều hướng 5 trang chức năng
│   └── Thẻ trạng thái hệ thống: Mô hình sẵn sàng, Dữ liệu 3,200 ảnh, CPU-only tối ưu
│
├── 🏠 Trang 1: Tổng quan (Dashboard)
│   ├── Hero Banner với huy hiệu nghiên cứu
│   ├── 4 Thẻ KPI thống kê số liệu thực (16 Lớp, 3,200 Ảnh, SVM 71.41%, RF 62.34%)
│   ├── Biểu đồ cột nhóm so sánh đa chiều (Accuracy, Precision, Recall, F1)
│   ├── Biểu đồ so sánh thời gian huấn luyện
│   ├── Bảng phân tích tóm tắt thực nghiệm
│   └── Nút Call-To-Action (CTA) "Thử phân loại ảnh ngay →"
│
├── 🔍 Trang 2: Phân loại trái cây (Chức năng trung tâm)
│   ├── Khung upload kéo thả hỗ trợ JPG, PNG, WEBP
│   ├── Bộ chọn ảnh mẫu 1-Click (Quick Test Samples) từ dataset thực tế
│   ├── Khung ảnh Preview kèm thông số kích thước, định dạng
│   ├── Trực quan hóa bản đồ gradient HOG Map 64x64
│   ├── Card kết quả SVM: Tên tiếng Việt, Trạng thái (🟢 Tươi / 🔴 Hỏng), Confidence %, Top 3 xác suất, Độ trễ (ms)
│   ├── Card kết quả RF: Tên tiếng Việt, Trạng thái, Confidence %, Top 3 xác suất, Độ trễ (ms)
│   ├── Banner đối chiếu: Thông báo Đồng thuận (Consensus) hoặc Bất đồng quan điểm (Disagreement)
│   └── Bảng lịch sử các lượt thử nghiệm trong phiên làm việc
│
├── 📊 Trang 3: So sánh mô hình (Academic Comparison)
│   ├── Bảng tổng hợp số liệu thực nghiệm đầy đủ kèm cột chênh lệch (Delta)
│   ├── Biểu đồ so sánh 4 chỉ số và biểu đồ thời gian
│   ├── Tab chuyển đổi xem Ma trận nhầm lẫn (Confusion Matrix) 16 lớp cho SVM và Random Forest
│   └── Chuyên mục phân tích học thuật lý giải nguyên nhân toán học
│
├── 📁 Trang 4: Dữ liệu và đánh giá (Dataset & Pipeline)
│   ├── Thống kê tổng quan bộ dữ liệu (3,200 mẫu gốc, phân chia 80/20)
│   ├── Bảng danh mục chi tiết 16 lớp trái cây (Song ngữ Anh - Việt)
│   └── 3 Khối mô tả quy trình kỹ thuật: Tiền xử lý, Trích xuất HOG, Huấn luyện mô hình
│
└── 📖 Trang 5: Giới thiệu đề tài (Project Overview & FAQ)
    ├── Sơ đồ kiến trúc xử lý tuần tự (End-to-End Pipeline ASCII Diagram)
    ├── Tóm lược cơ sở lý thuyết toán học của HOG, SVM và Random Forest
    └── Bộ câu hỏi vấn đáp thường gặp (FAQ) phục vụ thuyết trình và bảo vệ đề tài
```

---

## 🛠️ 6. CÁC LỖI KỸ THUẬT ĐÃ PHÁT HIỆN VÀ XỬ LÝ TRIỆT ĐỂ

1.  **Khắc phục lỗi thư mục lồng nhau khi giải nén:**
    *   *Hiện tượng:* Thư mục ảnh bị lồng dạng `data/Original Image/Original Image/`.
    *   *Xử lý:* Tự động làm phẳng và chuẩn hóa về cấu trúc phẳng `data/Original Image/<16 classes>`.
2.  **Sửa lỗi hiển thị ảnh HOG trên Streamlit:**
    *   *Lỗi gặp phải:* `streamlit.errors.StreamlitAPIException: Data is outside [0.0, 1.0] and clamp is not set.`
    *   *Nguyên nhân:* Mảng `hog_image` do `skimage.feature.hog` sinh ra có kiểu `float64` với các giá trị tích lũy vượt quá 1.0 (dao động trong khoảng $[0.0, 1.67]$).
    *   *Giải pháp tối ưu:* Áp dụng chuẩn hóa Min-Max toàn dải và chuyển sang định dạng số nguyên không dấu **`uint8` trong đoạn `[0, 255]`**:
        ```python
        h_min = float(np.min(hog_image))
        h_max = float(np.max(hog_image))
        if h_max > h_min:
            hog_display = np.clip((hog_image - h_min) / (h_max - h_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            hog_display = np.zeros(hog_image.shape, dtype=np.uint8)
        st.image(hog_display, use_container_width=True, caption="Biểu diễn phân bố hướng và mật độ gradient của biên cạnh (64x64)")
        ```
    *   *Kết quả:* Ảnh HOG hiển thị sắc nét, các đường biên gradient nổi bật rõ ràng, không còn lỗi runtime và không làm thay đổi vector đặc trưng 1,764 chiều của mô hình.
3.  **Xử lý cảnh báo Scikit-Learn và chuỗi LaTeX trong Python:**
    *   Khắc phục cảnh báo `FutureWarning` của `SVC(probability=True)`.
    *   Sử dụng chuỗi ký tự thô `r"""` cho các đoạn chứa ký hiệu toán học LaTeX (`\exp`, `\gamma`) trong `app.py`, loại bỏ hoàn toàn các cảnh báo `SyntaxWarning: invalid escape sequence`.

---

## 📦 7. CẤU TRÚC TỆP TIN TRONG THƯ MỤC DỰ ÁN

```text
d:\fruit-classification-ml\
├── data/                               # Dữ liệu ảnh trái cây
│   ├── Original Image/                 # 3,200 ảnh gốc (16 lớp x 200 ảnh cân bằng)
│   ├── Augmented Image/                # 12,335 ảnh tăng cường dữ liệu
│   └── README.md
├── models/                             # Tệp lưu trữ mô hình sau huấn luyện (.joblib)
│   ├── svm_model.joblib                # Mô hình SVM đã huấn luyện (~36 MB)
│   ├── rf_model.joblib                 # Mô hình Random Forest đã huấn luyện (~26 MB)
│   └── labels.joblib                   # Danh sách 16 nhãn chuẩn hóa
├── results/                            # Kết quả đánh giá và biểu đồ học thuật
│   ├── evaluation_metrics.json         # Toàn bộ số liệu Accuracy, F1, thời gian huấn luyện
│   ├── model_comparison.png            # Biểu đồ so sánh cột chất lượng
│   ├── confusion_matrix_svm.png        # Ma trận nhầm lẫn SVM (16 lớp)
│   ├── confusion_matrix_rf.png         # Ma trận nhầm lẫn Random Forest (16 lớp)
│   └── classification_report.txt       # Báo cáo chi tiết Precision, Recall từng lớp
├── src/                                # Module tiền xử lý và trích xuất đặc trưng
│   ├── __init__.py
│   └── features.py                     # Tiền xử lý PIL draft, trích xuất HOG đa luồng
├── app.py                              # Ứng dụng web Dashboard đa trang (Streamlit)
├── train_model.py                      # Script huấn luyện, đánh giá và xuất kết quả
├── requirements.txt                    # Thư viện phụ thuộc của dự án
├── README.md                           # Hướng dẫn kỹ thuật tổng quan
└── BAO_CAO_DU_AN.md                    # Báo cáo tổng kết toàn bộ quá trình thực hiện
```

---

## 🚀 8. HƯỚNG DẪN KHỞI CHẠY VÀ SỬ DỤNG HỆ THỐNG

### Bước 1: Mở Terminal trong Antigravity IDE
Sử dụng phím tắt ``Ctrl + ` `` hoặc mở menu **Terminal** $\rightarrow$ **New Terminal**.

### Bước 2: Huấn luyện lại mô hình (Nếu cần cập nhật lại mô hình)
```powershell
python train_model.py
```
*Script sẽ tự động đọc 3,200 ảnh, trích xuất đặc trưng HOG, huấn luyện SVM & Random Forest và lưu biểu đồ kết quả vào `results/`.*

### Bước 3: Khởi động giao diện người dùng Web Demo (Streamlit)
```powershell
streamlit run app.py
```
*Trình duyệt web sẽ tự động mở tại địa chỉ `http://localhost:8501`. Tại đây bạn có thể chuyển đổi giữa 5 trang chức năng, tải ảnh lên để nhận diện hoặc bấm chọn nhanh ảnh mẫu có sẵn để kiểm tra.*
