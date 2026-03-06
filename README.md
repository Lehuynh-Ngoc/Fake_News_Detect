# Hệ Thống Nhận Diện Tin Giả Tiếng Việt (Fake News Detection)

## Phần 1. TỔNG QUAN
Đề tài nghiên cứu và xây dựng hệ thống tự động nhận diện tin giả (Fake News) trong môi trường ngôn ngữ tiếng Việt. Với sự bùng nổ của mạng xã hội, tin giả trở thành một vấn đề cấp thiết, gây ra nhiều hệ lụy tiêu cực đến nhận thức và trật tự xã hội. Dự án này ứng dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học máy để phân loại tin tức một cách khách quan.

*   **Nhiệm vụ đồ án:**
    *   **Tính cấp thiết:** Đáp ứng nhu cầu lọc bỏ thông tin sai lệch trên không gian mạng.
    *   **Mục tiêu:** Xây dựng mô hình phân loại tin tức với độ chính xác cao và triển khai giao diện người dùng thân thiện.
    *   **Đối tượng & Phạm vi:** Các bài báo, tin tức mạng xã hội bằng tiếng Việt. Giới hạn ở việc phân loại nhị phân (Tin thật / Tin giả).
*   **Cấu trúc đồ án:**
    1.  **Chương 1 - Thu thập và Tiền xử lý:** Tập trung vào việc chuẩn hóa văn bản tiếng Việt.
    2.  **Chương 2 - Xây dựng Mô hình:** Áp dụng thuật toán Logistic Regression và TF-IDF.
    3.  **Chương 3 - Phát triển Hệ thống:** Xây dựng API bằng FastAPI và Frontend bằng React.
    4.  **Chương 4 - Đánh giá:** Phân tích các chỉ số Precision, Recall và F1-score.

## Phần 2. CƠ SỞ LÝ THUYẾT
Hệ thống được xây dựng trên nền tảng Python với các thư viện hiện đại:
*   **Ngôn ngữ & Framework:** 
    *   **Backend:** FastAPI (Python) - Hiệu suất cao, hỗ trợ tài liệu API tự động.
    *   **Frontend:** React (TypeScript) + Vite + TailwindCSS - Giao diện mượt mà, phản hồi nhanh.
*   **Xử lý ngôn ngữ tự nhiên (NLP):**
    *   Sử dụng thư viện `underthesea` để tách từ (Word Segmentation) chuyên sâu cho tiếng Việt.
    *   Sử dụng Regex để loại bỏ ký tự đặc biệt và chuẩn hóa văn bản.
*   **Học máy (Machine Learning):**
    *   **TF-IDF Vectorizer:** Biến đổi văn bản thành vector số dựa trên tần suất từ.
    *   **Logistic Regression:** Mô hình phân loại hiệu quả cho các bài toán văn bản quy mô vừa và nhỏ.
*   **Cấu trúc thư mục:**
    *   `main.py`: File thực thi chính cho API Server.
    *   `src/preprocess.py`: Chứa các hàm xử lý sạch dữ liệu đầu vào.
    *   `src/train.py`: Script huấn luyện mô hình từ dữ liệu thô.
    *   `data/`: Chứa file dữ liệu huấn luyện `fake_news.csv`.
    *   `models/`: Lưu trữ mô hình `fake_news_model.pkl` đã được đóng gói.
    *   `frontend/`: Mã nguồn giao diện người dùng.

## Phần 3. KẾT LUẬN VÀ KIẾN NGHỊ
*   **Kết luận:** Dự án đã hoàn thành mục tiêu đề ra, xây dựng được một quy trình khép kín từ tiền xử lý dữ liệu đến triển khai ứng dụng thực tế. Mô hình đạt kết quả khả quan trên tập dữ liệu kiểm tra.
*   **Kiến nghị:** Trong tương lai, hệ thống có thể nâng cấp bằng cách sử dụng các mô hình học sâu (Deep Learning) như BERT (PhoBERT) để hiểu ngữ nghĩa sâu hơn và mở rộng quy mô dữ liệu đa nguồn.

## Phần 4. THÔNG TIN THÀNH VIÊN
*   **Thành viên thực hiện:** Lê Huỳnh Ngọc
*   **Giáo viên hướng dẫn:** Phạm Thế Anh Phú
*   **Tiến độ:** 
    *   Tuần 1-2: Tìm hiểu lý thuyết và thu thập dữ liệu (100%).
    *   Tuần 3: Xây dựng mô hình và tiền xử lý (100%).
    *   Tuần 4: Phát triển Frontend/Backend (100%).
    *   Tuần 5: Kiểm thử và đóng gói (100%).
*   **Lần cập nhật gần nhất:** 06/03/2026

**Danh sách Tài liệu Tham khảo:**
1.  [Underthesea NLP Library](https://github.com/undertheseanlp/underthesea) - Truy cập 05/03/2026.
2.  [FastAPI Documentation](https://fastapi.tiangolo.com/) - Truy cập 01/03/2026.
3.  [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - Truy cập 02/03/2026.
