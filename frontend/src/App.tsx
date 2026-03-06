import React, { useState } from 'react'
import axios from 'axios'

interface PredictionResult {
  text: string;
  prediction: string;
  confidence: number;
}

function App() {
  const [newsText, setNewsText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState('')

  const handleCheck = async () => {
    if (!newsText.trim()) {
      setError('Vui lòng nhập nội dung tin tức cần kiểm tra.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      // Assuming backend is at http://localhost:8000
      const response = await axios.post('http://localhost:8000/predict', {
        text: newsText
      })
      setResult(response.data)
    } catch (err: any) {
      console.error('API Error:', err)
      setError('Đã có lỗi xảy ra khi kết nối tới máy chủ. Vui lòng thử lại sau.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-extrabold text-blue-600 mb-2">
            Hệ Thống Kiểm Chứng Tin Tức Giả
          </h1>
          <p className="text-slate-600">
            Sử dụng trí tuệ nhân tạo để nhận diện tin tức tiếng Việt.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-slate-200">
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Nội dung tin tức:
          </label>
          <textarea
            rows={8}
            className="block w-full rounded-lg border-slate-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-slate-700 p-4 border outline-none"
            placeholder="Dán nội dung bài báo hoặc đoạn tin tức cần kiểm tra vào đây..."
            value={newsText}
            onChange={(e) => setNewsText(e.target.value)}
          />
          
          {error && (
            <p className="mt-2 text-sm text-red-600">{error}</p>
          )}

          <div className="mt-6 flex justify-center">
            <button
              onClick={handleCheck}
              disabled={loading}
              className={`
                inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-full shadow-sm text-white 
                ${loading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} 
                focus:outline-none transition-all duration-200
              `}
            >
              {loading ? 'Đang xử lý...' : 'Kiểm tra tin tức'}
            </button>
          </div>
        </div>

        {/* Result Section */}
        {result && (
          <div className={`rounded-xl shadow-lg p-8 border-l-8 ${
            result.prediction === 'Fake' ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <h2 className={`text-2xl font-bold ${
                result.prediction === 'Fake' ? 'text-red-700' : 'text-green-700'
              }`}>
                {result.prediction === 'Fake' ? '⚠️ CẢNH BÁO: TIN GIẢ' : '✅ TIN THẬT'}
              </h2>
              <span className={`px-4 py-1 rounded-full text-sm font-semibold ${
                result.prediction === 'Fake' ? 'bg-red-200 text-red-800' : 'bg-green-200 text-green-800'
              }`}>
                Độ tin cậy: {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            
            <p className="text-slate-700 italic">
              "Dựa trên phân tích nội dung, hệ thống dự đoán đây là {result.prediction === 'Fake' ? 'tin giả hoặc tin sai sự thật' : 'tin tức có độ xác thực cao'}."
            </p>
            
            <div className="mt-6 text-sm text-slate-500">
              * Kết quả chỉ mang tính tham khảo. Hãy luôn kiểm chứng từ các nguồn tin chính thống.
            </div>
          </div>
        )}

        {/* Footer Info */}
        <div className="mt-16 text-center text-slate-400 text-sm">
          <p>© 2026 Vietnamese Fake News Detection Project</p>
        </div>
      </div>
    </div>
  )
}

export default App