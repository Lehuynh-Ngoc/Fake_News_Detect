import React, { useState } from 'react'
import axios from 'axios'

interface PredictionResult {
  text: string;
  prediction: string;
  warning_confidence: number;
  news_reliability: number;
  analysis: {
    summary: string;
    details: string[];
    patterns_found: string[];
    is_fact_check: boolean;
  };
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
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
              <h2 className={`text-2xl font-bold ${
                result.prediction === 'Fake' ? 'text-red-700' : 'text-green-700'
              }`}>
                {result.prediction === 'Fake' ? '⚠️ CẢNH BÁO: TIN GIẢ' : '✅ TIN THẬT'}
              </h2>
              <div className="flex flex-col gap-2">
                <span className={`px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                  result.prediction === 'Fake' ? 'bg-red-200 text-red-800' : 'bg-green-200 text-green-800'
                }`}>
                  Độ chính xác cảnh báo: {(result.warning_confidence * 100).toFixed(1)}%
                </span>
                <span className={`px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                  result.news_reliability < 0.5 ? 'bg-orange-200 text-orange-800' : 'bg-blue-200 text-blue-800'
                }`}>
                  Độ tin cậy của tin: {(result.news_reliability * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-white/50 p-5 rounded-lg border border-slate-200">
                <h3 className="text-lg font-bold text-slate-800 mb-3 flex items-center">
                  <span className="mr-2">🔍</span> Phân tích chi tiết từ AI
                </h3>
                <p className="text-slate-700 font-medium mb-4 pb-3 border-b border-slate-200">
                  {result.analysis.summary}
                </p>
                <ul className="space-y-3">
                  {result.analysis.details.map((detail, i) => (
                    <li key={i} className="flex items-start text-sm text-slate-600">
                      <span className="text-blue-500 mr-2">•</span>
                      {detail}
                    </li>
                  ))}
                </ul>
              </div>

              {result.analysis.patterns_found.length > 0 && (
                <div className="bg-red-100/50 p-5 rounded-lg border border-red-200">
                  <h3 className="text-sm font-bold text-red-800 mb-3 uppercase tracking-wider">
                    Các dấu hiệu nghi vấn cụ thể:
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.analysis.patterns_found.map((p, i) => (
                      <span key={i} className="bg-white px-3 py-1 rounded border border-red-200 text-xs text-red-700 font-medium">
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 text-sm text-slate-500 pt-4 border-t border-slate-200">
              * Kết quả được đánh giá bởi AI. Hãy luôn kiểm chứng từ các nguồn tin chính thống.
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