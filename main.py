from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import sys

# Import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocess import clean_text

app = FastAPI(title="Vietnamese Fake News Detection API")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Warning: Model not found. Please train the model first.")
except Exception as e:
    print(f"Error loading model: {e}")

class NewsItem(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(item: NewsItem):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    cleaned_text = clean_text(item.text)
    print(f"DEBUG: Input: {item.text[:100]}...")
    
    # Get probabilities
    probs = model.predict_proba([cleaned_text])[0]
    # Assuming model classes are [0, 1] for [Real, Fake]
    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    
    # HEURISTIC: Extensively expanded categorized suspicious patterns
    scam_categories = {
        "health_hoax": [
            "tiêu diệt hoàn toàn tế bào ung thư", "nước chanh nóng", "chữa khỏi bệnh", 
            "bí mật để bán thuốc", "thần dược", "chữa bách bệnh", "thuốc nam gia truyền", 
            "khỏi hẳn sau 7 ngày", "không cần phẫu thuật", "bác sĩ bệnh viện lớn dấu kín",
            "bài thuốc lạ", "tế bào gốc chữa mọi bệnh"
        ],
        "apocalypse_hoax": [
            "3 ngày bóng tối", "bóng tối hoàn toàn", "hệ thống điện ngừng hoạt động", 
            "tích trữ thực phẩm", "hiện tượng vũ trụ hiếm gặp", "ngày tận thế", 
            "thảm họa diệt vong", "tiểu hành tinh sắp va chạm", "người ngoài hành tinh xâm chiếm"
        ],
        "get_rich_quick": [
            "dự đoán chính xác 100%", "kết quả xổ số", "giàu có chỉ sau vài ngày", 
            "phần mềm ai dự đoán", "bí quyết làm giàu", "kiếm tiền tại nhà dễ dàng", 
            "vốn ít lời nhiều", "cam kết lợi nhuận", "không làm cũng có ăn",
            "nhận lương theo ngày", "việc nhẹ lương cao"
        ],
        "finance_crypto_scam": [
            "sàn quốc tế uy tín", "nhận quà tặng tri ân", "tiền ảo sắp lên sàn", 
            "nhân đôi tài sản", "đầu tư bao lỗ", "lãi suất 30% mỗi tháng", 
            "liên kết với ngân hàng", "nạp tiền nhận hoa hồng", "cơ hội nghìn năm có một"
        ],
        "recruitment_scam": [
            "tuyển cộng tác viên xử lý đơn hàng", "shopee tuyển dụng", "tiki tuyển dụng", 
            "việc làm online không cần bằng cấp", "không mất phí", "đặt cọc giữ chỗ"
        ],
        "tech_battery_hoax": [
            "pin phát nổ", "nạp điện liên tục suốt đêm", "nổ tung", "sạc qua đêm gây cháy",
            "tuyệt đối không nên cắm sạc khi ngủ", "nổ sau vài tuần sử dụng", "hỏng pin hoàn toàn"
        ],
        "conspiracy": [
            "bí mật bị che giấu", "bí mật bị giấu kín", "không cho bạn biết", 
            "sự thật kinh hoàng", "phát minh chấn động", "tổ chức ngầm", 
            "thông tin bị cấm", "không thể tin nổi", "sự thật đằng sau"
        ]
    }
    
    # DEBUNKING KEYWORDS: Words that indicate a fact-check or correction
    debunking_keywords = [
        "không có bằng chứng", "tin đồn", "sai sự thật", "bác bỏ", "đính chính", 
        "cảnh báo về", "không đúng", "giả mạo", "nasa khẳng định", "khoa học chứng minh",
        "không thể dự đoán", "không thể", "không có khả năng", "ngẫu nhiên hoàn toàn", 
        "chưa có nghiên cứu", "chuyên gia khẳng định", "tuy nhiên", "thực tế là",
        "bộ công an cảnh báo", "khuyến cáo người dân", "lừa đảo chiếm đoạt tài sản",
        "người dân cần cảnh giác", "phản bác thông tin", "theo thông tin từ bộ", 
        "cơ quan chức năng xác nhận", "kiểm chứng thông tin", "vạch trần",
        "mạch quản lý pin", "tự động ngắt", "chế độ sạc duy trì", "không gây nguy hiểm",
        "theo khuyến nghị của hãng", "sạc chính hãng", "cơ chế hoạt động", "an toàn sử dụng",
        "samsung", "apple", "người dùng yên tâm", "bảo vệ tuổi thọ pin"
    ]
    
    heuristic_boost = 0.0
    text_lower = item.text.lower()
    
    # Check for debunking signals first
    is_debunking = any(keyword in text_lower for keyword in debunking_keywords)
    
    found_patterns = []
    for category, patterns in scam_categories.items():
        for pattern in patterns:
            if pattern in text_lower:
                # If it's a debunking article, reduce the penalty significantly
                boost = 0.25 if not is_debunking else 0.05
                heuristic_boost += boost
                found_patterns.append(pattern)
            
    # Absolute claims check
    if ("100%" in item.text or "chính xác tuyệt đối" in text_lower) and not is_debunking:
        heuristic_boost += 0.3
        
    # If it's identified as debunking, we might even give a "Real" bias
    if is_debunking and heuristic_boost > 0:
        print(f"DEBUG: Debunking detected, mitigating boost from {heuristic_boost:.2f}")
        heuristic_boost = max(0, heuristic_boost - 0.2)
        
    # Calculate final fake probability
    final_fake_prob = min(0.99, fake_prob + heuristic_boost)
    
    # Determine Label
    label = "Fake" if final_fake_prob >= 0.4 else "Real"
    
    # Generate Detailed Analysis
    analysis_report = []
    
    # 1. Base ML Analysis
    if fake_prob > 0.7:
        analysis_report.append("Mô hình học máy phát hiện cấu trúc ngôn ngữ rất giống với các mẫu tin giả đã biết (xác suất thống kê cao).")
    elif fake_prob < 0.3:
        analysis_report.append("Cấu trúc câu từ và cách trình bày tương đồng với phong cách báo chí chính thống.")
    else:
        analysis_report.append("Ngôn ngữ bài viết nằm ở mức trung lập, có sự pha trộn giữa các đặc điểm tin tức và nội dung tự do.")

    # 2. Heuristic Analysis
    if found_patterns:
        analysis_report.append(f"Phát hiện {len(found_patterns)} dấu hiệu nội dung nghi vấn thuộc nhóm: {', '.join(set([k for k, v in scam_categories.items() if any(p in found_patterns for p in v)]))}.")
    
    # 3. Context & Debunking Analysis
    if is_debunking:
        analysis_report.append("Hệ thống nhận diện được các tín hiệu đính chính hoặc phản biện (như 'không có bằng chứng', 'tin đồn'). Đây là cơ sở quan trọng để giảm mức độ cảnh báo.")
    
    if "100%" in item.text or "chính xác tuyệt đối" in text_lower:
        if not is_debunking:
            analysis_report.append("Bài viết sử dụng các khẳng định tuyệt đối (100%), thường là dấu hiệu của việc cường điệu hóa thông tin.")

    # Conclusion Reasoning
    if label == "Fake":
        reasoning = "Cảnh báo dựa trên sự kết hợp giữa các từ khóa nhạy cảm và cấu trúc câu mang tính chất gây hoang mang hoặc hứa hẹn phi thực tế."
    else:
        reasoning = "Nội dung được đánh giá là tin cậy nhờ vào cách tiếp cận khách quan hoặc có sự xuất hiện của các từ khóa kiểm chứng khoa học/chính thống."

    # Calculate final metrics
    warning_confidence = final_fake_prob if label == "Fake" else (1.0 - final_fake_prob)
    news_reliability = 1.0 - final_fake_prob

    print(f"DEBUG: Prediction: {label} (Warning Conf: {warning_confidence:.2f}, News Reliability: {news_reliability:.2f})")
    
    return {
        "text": item.text,
        "prediction": label,
        "warning_confidence": float(warning_confidence),
        "news_reliability": float(news_reliability),
        "analysis": {
            "summary": reasoning,
            "details": analysis_report,
            "patterns_found": found_patterns,
            "is_fact_check": is_debunking
        }
    }

# Serve Frontend (after build)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)