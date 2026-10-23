"""
features.py - Module tiền xử lý ảnh và trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
Phục vụ đề tài Phân loại trái cây bằng SVM và Random Forest.
Thiết kế tối ưu cho RAM 8GB, chạy mượt mà trên CPU và tương thích tốt với Windows.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image
from skimage.feature import hog


# Các định dạng ảnh được hỗ trợ
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_safely(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Đọc ảnh an toàn bằng PIL và chuyển sang numpy array (RGB).
    Khắc phục triệt để lỗi đường dẫn chứa dấu tiếng Việt hoặc khoảng trắng trên Windows
    mà cv2.imread thường hay gặp phải.
    """
    try:
        path_obj = Path(image_path)
        if not path_obj.exists():
            return None
        with Image.open(path_obj) as img:
            # Chuyển đổi sang hệ màu RGB (loại bỏ kênh alpha nếu có)
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)
    except Exception as e:
        print(f"[Cảnh báo] Không thể đọc ảnh: {image_path}. Chi tiết lỗi: {e}")
        return None


def preprocess_image(
    image: Union[np.ndarray, Image.Image, str, Path],
    target_size: Tuple[int, int] = (64, 64)
) -> Optional[np.ndarray]:
    """
    Tiền xử lý ảnh:
    1. Đọc và chuẩn hóa ảnh về numpy array RGB / Grayscale.
    2. Resize về kích thước target_size (mặc định 64x64 giúp tối ưu bộ nhớ RAM 8GB).
    3. Chuyển sang ảnh mức xám (grayscale) để trích xuất HOG hiệu quả và nhanh chóng.
    
    Args:
        image: Đường dẫn ảnh, đối tượng PIL Image hoặc numpy array.
        target_size: Kích thước chuẩn hóa (width, height), mặc định (64, 64).
        
    Returns:
        np.ndarray: Ảnh mức xám 2D [height, width] với giá trị uint8.
    """
    # 1. Tối ưu mở ảnh trực tiếp từ đường dẫn bằng PIL draft (tăng tốc gấp 7x cho ảnh độ phân giải cao)
    if isinstance(image, (str, Path)):
        try:
            path_obj = Path(image)
            if not path_obj.exists():
                return None
            with Image.open(path_obj) as img:
                img.draft("L", target_size)
                resized = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
                return np.array(resized)
        except Exception:
            # Fallback nếu gặp định dạng đặc biệt
            img_arr = read_image_safely(image)
            if img_arr is None:
                return None
    elif isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_arr = image.copy()
        if img_arr.ndim == 2:
            pass
        elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :3]
    else:
        return None

    # 2. Resize ảnh bằng PIL (chất lượng cao, tiết kiệm tài nguyên)
    pil_img = Image.fromarray(img_arr)
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # 3. Chuyển sang ảnh mức xám (L)
    gray_pil = resized_pil.convert("L")
    gray_arr = np.array(gray_pil)

    return gray_arr


def extract_hog_features(
    image: Union[np.ndarray, Image.Image, str, Path],
    target_size: Tuple[int, int] = (64, 64),
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Trích xuất vector đặc trưng HOG từ ảnh.
    
    Tham số cấu hình HOG:
    - orientations: Số lượng hướng gradient (thường chọn 9 cho kết quả cân bằng).
    - pixels_per_cell: Kích thước mỗi ô (8x8 pixel).
    - cells_per_block: Số cell trong một block chuẩn hóa (2x2 cell).
    - visualize: Nếu True, trả về cả ảnh minh họa HOG (rất hữu ích cho giao diện Streamlit).
    
    Returns:
        - Nếu visualize=False: Trả về vector 1D đặc trưng HOG (dtype float32).
        - Nếu visualize=True: Trả về tuple (hog_feature_vector, hog_visualization_image).
    """
    gray_image = preprocess_image(image, target_size=target_size)
    if gray_image is None:
        return (None, None) if visualize else None

    if visualize:
        features, hog_image = hog(
            gray_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            visualize=True
        )
        return features.astype(np.float32), hog_image
    else:
        features = hog(
            gray_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            visualize=False
        )
        return features.astype(np.float32)


def load_dataset_features(
    data_dir: Union[str, Path] = "data",
    target_size: Tuple[int, int] = (64, 64),
    max_samples_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Duyệt cây thư mục dữ liệu data/, trích xuất đặc trưng HOG cho từng ảnh trái cây.
    Tự động nhận diện cấu trúc thư mục 'Original Image' hoặc các thư mục lớp trực tiếp.
    Sử dụng xử lý đa luồng (multi-threading) để trích xuất cực nhanh.
    
    Returns:
        X (np.ndarray): Mảng 2D kích thước (N_samples, N_features), float32.
        y (np.ndarray): Mảng 1D nhãn số nguyên tương ứng từng ảnh.
        label_names (List[str]): Danh sách tên các loại trái cây theo thứ tự chỉ số.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    base_path = Path(data_dir)

    if not base_path.exists():
        print(f"[Thông báo] Thư mục dữ liệu '{data_dir}' chưa tồn tại.")
        return np.array([]), np.array([]), []

    # Tự động nhận diện nếu dữ liệu nằm trong 'Original Image'
    candidate_sub = base_path / "Original Image"
    if candidate_sub.exists() and candidate_sub.is_dir():
        search_path = candidate_sub
    else:
        search_path = base_path

    # Tìm các thư mục con tương ứng với từng loại quả
    class_dirs = [
        d for d in search_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in {"Augmented Image", "Original Image"}
    ]
    class_dirs.sort(key=lambda p: p.name)

    if not class_dirs:
        print(f"[Thông báo] Thư mục '{search_path}' hiện chưa có thư mục con nào chứa ảnh các loại trái cây.")
        return np.array([]), np.array([]), []

    label_names = [d.name for d in class_dirs]
    features_list = []
    labels_list = []

    print(f"Bắt đầu nạp dữ liệu từ '{search_path}' ({len(label_names)} lớp trái cây: {', '.join(label_names)})...")

    # Xác định số luồng tối ưu (tối đa 8 luồng để mượt CPU)
    num_workers = min(8, max(2, (os.cpu_count() or 4)))

    for class_idx, class_dir in enumerate(class_dirs):
        # Thu thập các file ảnh hợp lệ
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if max_samples_per_class and len(image_files) > max_samples_per_class:
            image_files = image_files[:max_samples_per_class]

        # Trích xuất song song các ảnh trong lớp
        def _extract_single(path):
            return extract_hog_features(path, target_size=target_size, visualize=False)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            feats = list(executor.map(_extract_single, image_files))

        valid_feats = [f for f in feats if f is not None]
        features_list.extend(valid_feats)
        labels_list.extend([class_idx] * len(valid_feats))

        print(f" -> Lớp '{class_dir.name}': Đã trích xuất thành công {len(valid_feats)}/{len(image_files)} ảnh.")

    if not features_list:
        print("[Thông báo] Không tìm thấy ảnh hợp lệ nào trong các thư mục lớp.")
        return np.array([]), np.array([]), label_names

    X = np.asarray(features_list, dtype=np.float32)
    y = np.asarray(labels_list, dtype=np.int64)

    print(f"Tổng hợp hoàn tất: {X.shape[0]} mẫu dữ liệu, mỗi mẫu có {X.shape[1]} đặc trưng HOG.")
    return X, y, label_names
