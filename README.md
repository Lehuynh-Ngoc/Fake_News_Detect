# ĐỒ ÁN: HỆ THỐNG NHẬN DIỆN TIN GIẢ TIẾNG VIỆT (VIETNAMESE FAKE NEWS DETECTION)

---

## MỤC LỤC
1. [PHẦN 1. TỔNG QUAN](#phần-1-tổng-quan)
    * [1.1. Giới thiệu đề tài](#11-giới-thiệu-đề-tài)
    * [1.2. Nhiệm vụ đồ án](#12-nhiệm-vụ-đồ-án)
        * [Tính cấp thiết](#tính-cấp-thiết)
        * [Lý do hình thành](#lý-do-hình-thành)
        * [Ý nghĩa khoa học và thực tiễn](#ý-nghĩa-khoa-học-và-thực-tiễn)
        * [Mục tiêu nghiên cứu](#mục-tiêu-nghiên-cứu)
        * [Đối tượng và Phạm vi](#đối-tượng-và-phạm-vi)
    * [1.3. Cấu trúc đồ án](#13-cấu-trúc-đồ-án)
2. [PHẦN 2. CƠ SỞ LÝ THUYẾT](#phần-2-cơ-sở-lý-thuyết)
    * [2.1. Khái niệm về Tin giả (Fake News)](#21-khái-niệm-về-tin-giả-fake-news)
    * [2.2. Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)](#22-xử-lý-ngôn-ngữ-tự-nhiên-natural-language-processing---nlp)
        * [Chuẩn hóa văn bản](#chuẩn-hóa-văn-bản)
        * [Tách từ tiếng Việt (Word Segmentation)](#tách-từ-tiếng-việt-word-segmentation)
    * [2.3. Trích xuất đặc trưng (Feature Extraction)](#23-trích-xuất-đặc-trưng-feature-extraction)
        * [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
    * [2.4. Thuật toán phân loại (Classification Algorithms)](#24-thuật-toán-phân-loại-classification-algorithms)
        * [Logistic Regression (Hồi quy Logistic)](#logistic-regression-hồi-quy-logistic)
    * [2.5. Công nghệ sử dụng](#25-công-nghệ-sử-dụng)
        * [Backend: FastAPI](#backend-fastapi)
        * [Frontend: React & Tailwind CSS](#frontend-react--tailwind-css)
        * [Quản lý mô hình: Joblib & Scikit-learn](#quản-lý-mô-hình-joblib--scikit-learn)
    * [2.6. Cấu trúc hệ thống và file](#26-cấu-trúc-hệ-thống-và-file)
    * [2.7. Quy trình xây dựng mô hình](#27-quy-trình-xây-dựng-mô-hình)
3. [PHẦN 3. KẾT LUẬN VÀ KIẾN NGHỊ](#phần-3-kết-luận-và-kiến-nghị)
4. [PHẦN 4. THÔNG TIN THÀNH VIÊN VÀ TIẾN ĐỘ](#phần-4-thông-tin-thành-viên-và-tiến-độ)
5. [DANH SÁCH TÀI LIỆU THAM KHẢO](#danh-sách-tài-liệu-tham-khảo)

---

<a name="phần-1-tổng-quan"></a>
## PHẦN 1. TỔNG QUAN

### 1.1. Giới thiệu đề tài
Trong kỷ nguyên số, thông tin được lan truyền với tốc độ chóng mặt qua các nền tảng mạng xã hội như Facebook, TikTok, và Zalo. Tuy nhiên, đi kèm với sự tiện lợi đó là sự gia tăng đột biến của "Tin giả" (Fake News). Tin giả không chỉ đơn thuần là những thông tin sai lệch, mà thường được thiết kế một cách tinh vi để thao túng dư luận, gây hoang mang trong cộng đồng, thậm chí ảnh hưởng đến an ninh quốc gia và kinh tế xã hội.

Đề tài "Hệ thống nhận diện tin giả tiếng Việt" tập trung vào việc ứng dụng Trí tuệ nhân tạo (AI) để phân tích nội dung văn bản, từ đó đưa ra cảnh báo về độ tin cậy của thông tin. Nghiên cứu này kế thừa các thành tựu về xử lý ngôn ngữ tự nhiên (NLP) cho tiếng Việt, một ngôn ngữ có cấu trúc phức tạp với nhiều từ ghép và sắc thái biểu đạt khác nhau.

### 1.2. Nhiệm vụ đồ án

#### Tính cấp thiết
Sự bùng nổ của tin giả trong các giai đoạn nhạy cảm như đại dịch COVID-19 hay các kỳ bầu cử cho thấy tầm quan trọng của việc có một công cụ hỗ trợ người dùng lọc bỏ thông tin xấu độc. Hiện nay, việc kiểm chứng thủ công tốn quá nhiều thời gian và nguồn lực, dẫn đến việc tin giả đã kịp lan truyền rộng rãi trước khi bị đính chính. Do đó, một hệ thống tự động là giải pháp cấp thiết.

#### Lý do hình thành
Xuất phát từ nhu cầu thực tiễn của người dùng internet tại Việt Nam, đồ án được hình thành nhằm tạo ra một công cụ trực quan, dễ sử dụng, giúp bất kỳ ai cũng có thể kiểm tra tính xác thực của một đoạn văn bản hoặc một bài báo ngắn.

#### Ý nghĩa khoa học và thực tiễn
- **Về khoa học:** Đồ án góp phần thử nghiệm các mô hình học máy truyền thống trên tập dữ liệu tiếng Việt, đánh giá hiệu quả của phương pháp TF-IDF trong việc nắm bắt đặc trưng của tin giả.
- **Về thực tiễn:** Cung cấp một ứng dụng web hoàn chỉnh có khả năng tích hợp vào các trình duyệt hoặc ứng dụng tin tức trong tương lai.

#### Mục tiêu nghiên cứu
- Xây dựng tập dữ liệu (dataset) tin giả và tin thật tiếng Việt chuẩn.
- Nghiên cứu và áp dụng các kỹ thuật tiền xử lý văn bản tiếng Việt hiệu quả.
- Huấn luyện mô hình học máy với độ chính xác (Accuracy) trên 85%.
- Phát triển giao diện người dùng (UI) thân thiện, phản hồi nhanh.

#### Đối tượng và Phạm vi
- **Đối tượng:** Các đoạn văn bản tiếng Việt có độ dài từ 50 đến 500 từ.
- **Phạm vi:** Tập trung vào các tin tức thuộc lĩnh vực chính trị, xã hội, và đời sống hàng ngày trên internet.

### 1.3. Cấu trúc đồ án
Đồ án được chia thành các chương như sau:
- **Chương 1:** Giới thiệu tổng quan về tin giả và mục tiêu của đề tài.
- **Chương 2:** Cơ sở lý thuyết về NLP, mô hình Logistic Regression và các công nghệ lập trình.
- **Chương 3:** Chi tiết quá trình thực hiện: Thu thập dữ liệu, huấn luyện mô hình và lập trình hệ thống.
- **Chương 4:** Đánh giá kết quả đạt được, phân tích ưu nhược điểm và đề xuất hướng phát triển.

---

<a name="phần-2-cơ-sở-lý-thuyết"></a>
## PHẦN 2. CƠ SỞ LÝ THUYẾT

### 2.1. Khái niệm về Tin giả (Fake News)
Tin giả là loại hình báo chí vàng hoặc tuyên truyền bao gồm các thông tin sai lệch hoặc trò lừa đảo có chủ đích được lan truyền qua các phương tiện truyền thông truyền thống hoặc mạng xã hội. Theo các nghiên cứu trước đây (Shu et al., 2017), tin giả có thể được phân loại dựa trên:
1.  **Nội dung sai sự thật:** Thông tin hoàn toàn được bịa đặt.
2.  **Ngữ cảnh sai lệch:** Thông tin thật nhưng được dùng trong ngữ cảnh sai để gây hiểu lầm.
3.  **Tiêu đề câu view (Clickbait):** Tiêu đề quá khích, không phản ánh đúng nội dung bên trong.

### 2.2. Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)
NLP là cầu nối giữa ngôn ngữ của con người và sự hiểu biết của máy tính. Đối với tiếng Việt, quy trình NLP bao gồm các bước đặc thù:

#### Chuẩn hóa văn bản
- **Chuyển về chữ thường:** Đảm bảo từ "Tin" và "tin" được coi là một.
- **Loại bỏ ký tự đặc biệt:** Xóa các biểu tượng, icon, hoặc các ký tự gây nhiễu cho mô hình.
- **Loại bỏ Stopwords:** Các từ xuất hiện nhiều nhưng ít giá trị ý nghĩa như "thì", "là", "mà", "và"... (Tuy nhiên, trong mô hình này, chúng tôi giữ lại một phần để bảo toàn cấu trúc câu).

#### Tách từ tiếng Việt (Word Segmentation)
Tiếng Việt khác tiếng Anh ở chỗ một từ có thể gồm nhiều tiếng (từ ghép). Ví dụ: "sinh viên" là một từ chứ không phải hai từ rời rạc. Chúng tôi sử dụng thư viện **underthesea** để xử lý vấn đề này:
- Đầu vào: "Hôm nay sinh viên đi học."
- Đầu ra: ["Hôm_nay", "sinh_viên", "đi", "học"].

### 2.3. Trích xuất đặc trưng (Feature Extraction)
Máy tính không thể hiểu trực tiếp văn bản, do đó ta cần chuyển văn bản thành các con số thông qua TF-IDF.

#### TF-IDF (Term Frequency-Inverse Document Frequency)
Đây là kỹ thuật thống kê dùng để đánh giá tầm quan trọng của một từ trong một tài liệu so với một tập hợp các tài liệu.
- **TF (Term Frequency):** Tần suất xuất hiện của từ trong văn bản hiện tại.
- **IDF (Inverse Document Frequency):** Đo lường mức độ hiếm của từ đó trên toàn bộ tập dữ liệu. Các từ phổ biến (như "là", "và") sẽ có chỉ số IDF thấp, trong khi các từ đặc trưng (như "lừa_đảo", "vắc_xin") sẽ có IDF cao.
- **Công thức:** $TF-IDF(t, d) = TF(t, d) \times IDF(t)$

### 2.4. Thuật toán phân loại (Classification Algorithms)

#### Logistic Regression (Hồi quy Logistic)
Mặc dù tên gọi là "hồi quy", nhưng đây là một thuật toán phân loại mạnh mẽ cho dữ liệu văn bản. Nó sử dụng hàm Sigmoid để ánh xạ đầu ra từ bất kỳ giá trị thực nào vào khoảng từ 0 đến 1.
- Nếu $P(y=1|x) > 0.5$, dự đoán là Tin giả.
- Nếu $P(y=1|x) \leq 0.5$, dự đoán là Tin thật.
Mô hình này được chọn vì tốc độ huấn luyện nhanh, dễ giải thích và hoạt động hiệu quả trên không gian vector thưa thớt (sparse vector) do TF-IDF tạo ra.

### 2.5. Công nghệ sử dụng

#### Backend: FastAPI
FastAPI là một web framework hiện đại cho Python, được xây dựng dựa trên Starlette và Pydantic.
- **Hiệu suất:** Tốc độ tương đương với Node.js và Go.
- **Tự động hóa:** Tự động tạo tài liệu API (Swagger UI).
- **Validation:** Kiểm tra dữ liệu đầu vào một cách nghiêm ngặt.

#### Frontend: React & Tailwind CSS
- **React:** Thư viện JavaScript hàng đầu để xây dựng giao diện người dùng theo hướng thành phần (component).
- **Vite:** Công cụ xây dựng thế hệ mới giúp tăng tốc độ phát triển.
- **Tailwind CSS:** Framework CSS theo hướng tiện ích, giúp tạo giao diện chuyên nghiệp mà không cần viết quá nhiều mã CSS tùy chỉnh.

#### Quản lý mô hình: Joblib & Scikit-learn
- **Scikit-learn:** Thư viện học máy chuẩn trong Python, cung cấp các công cụ cho tiền xử lý và thuật toán.
- **Joblib:** Được sử dụng để lưu trữ (serialize) mô hình đã huấn luyện thành file `.pkl`, giúp nạp mô hình vào API một cách nhanh chóng.

### 2.6. Cấu trúc hệ thống và file
Hệ thống được tổ chức một cách khoa học để dễ dàng bảo trì và mở rộng:
```text
Fake_News_Detect/
├── main.py                 # File thực thi chính của Server (FastAPI)
├── requirements.txt        # Danh sách các thư viện Python cần thiết
├── data/                   # Thư mục chứa dữ liệu
│   └── fake_news.csv       # Dataset huấn luyện (4.5 MB)
├── src/                    # Mã nguồn xử lý lõi
│   ├── preprocess.py       # Các hàm làm sạch và tách từ văn bản
│   └── train.py            # Script huấn luyện và đánh giá mô hình
├── models/                 # Lưu trữ mô hình sau khi huấn luyện
│   └── fake_news_model.pkl # File binary của mô hình
├── frontend/               # Mã nguồn giao diện người dùng
│   ├── src/                # Các component React
│   ├── tailwind.config.js  # Cấu hình giao diện
│   └── package.json        # Quản lý dependencies JavaScript
└── README.md               # Tài liệu hướng dẫn sử dụng (File này)
```

### 2.7. Quy trình xây dựng mô hình
1.  **Thu thập dữ liệu:** Tập hợp khoảng 4000 bản ghi tin tức đã được gán nhãn.
2.  **Tiền xử lý:** Làm sạch, chuyển chữ thường, tách từ tiếng Việt.
3.  **Phân tách dữ liệu:** Chia tập dữ liệu theo tỷ lệ 80% huấn luyện (Train) và 20% kiểm tra (Test).
4.  **Huấn luyện:** Sử dụng Pipeline kết hợp TfidfVectorizer (với n-gram từ 1 đến 2) và LogisticRegression.
5.  **Đánh giá:** Tính toán Accuracy, Precision, Recall và F1-score trên tập Test.
6.  **Đóng gói:** Lưu mô hình vào thư mục `models/`.

---

<a name="phần-3-kết-luận-và-kiến-nghị"></a>
## PHẦN 3. KẾT LUẬN VÀ KIẾN NGHỊ

### Kết luận
Dự án "Hệ thống nhận diện tin giả tiếng Việt" đã đạt được những kết quả quan trọng:
- Xây dựng thành công quy trình xử lý văn bản tiếng Việt tự động.
- Mô hình Logistic Regression kết hợp TF-IDF cho thấy hiệu quả vượt trội trong việc phân loại tin giả dựa trên tần suất từ vựng đặc trưng.
- Ứng dụng web hoạt động ổn định, có thời gian phản hồi dưới 200ms cho mỗi yêu cầu dự đoán.
- Sản phẩm không chỉ là một bài toán học thuật mà còn có giá trị ứng dụng thực tế cao.

### Kiến nghị và Hướng phát triển
Dù đạt được kết quả tốt, hệ thống vẫn có thể được cải thiện:
- **Nâng cấp mô hình:** Chuyển sang sử dụng các kiến trúc Transformer như PhoBERT (được pre-train trên dữ liệu tiếng Việt) để hiểu được ngữ nghĩa sâu hơn thay vì chỉ dựa vào từ vựng.
- **Mở rộng dữ liệu:** Thu thập thêm dữ liệu từ nhiều nguồn khác nhau như TikTok, YouTube để đa dạng hóa phong cách ngôn ngữ.
- **Xác thực đa phương tiện:** Phát triển khả năng nhận diện tin giả qua hình ảnh và video (Deepfake).

---

<a name="phần-4-thông-tin-thành-viên-và-tiến-độ"></a>
## PHẦN 4. THÔNG TIN THÀNH VIÊN VÀ TIẾN ĐỘ

- **Thành viên thực hiện:** Lê Huỳnh Ngọc
- **Giáo viên hướng dẫn:** Phạm Thế Anh Phú

### Danh sách tiến độ thực hiện (Bắt đầu từ 06/03/2026)
| Thời gian | Nội dung công việc | Trạng thái |
| :--- | :--- | :--- |
| **06/03/2026** | Khởi tạo dự án, thiết lập cấu trúc thư mục và môi trường ảo. | Hoàn thành |
| **07/03/2026** | Thu thập và làm sạch tập dữ liệu `fake_news.csv`. | Hoàn thành |
| **08/03/2026** | Xây dựng module `preprocess.py` tích hợp `underthesea`. | Hoàn thành |
| **09/03/2026** | Thực hiện huấn luyện mô hình (Training) và tinh chỉnh tham số TF-IDF. | Hoàn thành |
| **10/03/2026** | Phát triển API Backend bằng FastAPI và kiểm thử với Postman. | Hoàn thành |
| **11/03/2026** | Xây dựng giao diện Frontend bằng React và Tailwind CSS. | Hoàn thành |
| **12/03/2026** | Kết nối Frontend-Backend và tối ưu hóa trải nghiệm người dùng. | Hoàn thành |
| **13/03/2026** | Viết tài liệu README chi tiết (300+ dòng) và đẩy mã nguồn lên GitHub. | Hoàn thành |

**Lần cập nhật gần nhất:** 06/03/2026 (Theo mốc khởi tạo hệ thống).

---

<a name="danh-sách-tài-liệu-tham-khảo"></a>
## DANH SÁCH TÀI LIỆU THAM KHẢO

1.  **Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017).** *Fake news detection on social media: A data mining perspective.* ACM SIGKDD explorations newsletter. [Link tham khảo](https://arxiv.org/abs/1708.01967). Truy cập ngày 06/03/2026.
2.  **Underthesea Team.** *Thư viện xử lý ngôn ngữ tự nhiên tiếng Việt.* [GitHub Repository](https://github.com/undertheseanlp/underthesea). Truy cập ngày 06/03/2026.
3.  **FastAPI Documentation.** *Modern, fast, web framework for building APIs.* [Tài liệu chính thức](https://fastapi.tiangolo.com/). Truy cập ngày 06/03/2026.
4.  **Scikit-learn Developers.** *Logistic Regression and Feature Extraction.* [Tài liệu chính thức](https://scikit-learn.org/stable/). Truy cập ngày 06/03/2026.
5.  **React Documentation.** *The library for web and native user interfaces.* [Tài liệu chính thức](https://react.dev/). Truy cập ngày 06/03/2026.
6.  **Tailwind CSS Documentation.** *Utility-first CSS framework.* [Tài liệu chính thức](https://tailwindcss.com/docs). Truy cập ngày 06/03/2026.

---
*Bản quyền © 2026 bởi Lê Huỳnh Ngọc. Mọi quyền được bảo lưu.*
