import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

interface SystemStats {
  cpu: number;
  ram_total: number;
  ram_used: number;
  ram_percent: number;
  disk: number;
  gpus: Array<{
    name: string;
    usage: number;
    memory_total: number;
    memory_used: number;
  }>;
}

interface TrainingDetails {
  current: number;
  total: number;
  elapsed: string;
  remaining: string;
  speed: string;
}

interface TrainingResult {
  accuracy: number;
  f1: number;
  precision: number;
  recall: number;
  false_alarm_rate: number;
  duration: number;
  confusion_matrix: number[][];
}

const TrainingPage: React.FC = () => {
  const [modelType, setModelType] = useState('phobert');
  const [epochs, setEpochs] = useState(10);
  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [details, setDetails] = useState<TrainingDetails>({ current: 0, total: 0, elapsed: '00:00', remaining: '00:00', speed: '0it/s' });
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [results, setResults] = useState<Record<string, TrainingResult>>({});
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [connectionError, setConnectionError] = useState(false);
  
  const terminalRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Scroll terminal to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  // Poll system stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get('http://localhost:8001/system-stats');
        setSystemStats(response.data);
        setConnectionError(false);
      } catch (err) {
        console.error('Failed to fetch system stats', err);
        setConnectionError(true);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 3000);
    return () => clearInterval(interval);
  }, []);

  // Recover state on mount
  useEffect(() => {
    const recoverState = async () => {
      try {
        const response = await axios.get('http://localhost:8001/training-status');
        const data = response.data;
        if (data.is_training) {
          setLogs(data.logs);
          setProgress(data.progress);
          setDetails(data.details);
          setModelType(data.current_model);
          setIsTraining(true);
          setStatus('running');
          connectSSE();
        } else if (data.logs.length > 0) {
          setLogs(data.logs);
          setProgress(data.progress);
          setDetails(data.details);
          setStatus(data.progress === 100 ? 'completed' : 'idle');
        }
      } catch (err) {
        console.error('Failed to recover state', err);
      }
    };
    recoverState();
  }, []);

  const connectSSE = () => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    
    const eventSource = new EventSource('http://localhost:8001/training-events');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'log') {
        setLogs(prev => {
          if (prev.includes(data.content) && prev.indexOf(data.content) > prev.length - 5) return prev;
          return [...prev, data.content];
        });
        if (data.progress !== undefined) setProgress(data.progress);
        if (data.details) setDetails(data.details);
      } else if (data.type === 'status') {
        if (data.status === 'completed') {
          setProgress(100); setIsTraining(false); setStatus('completed'); eventSource.close();
        } else if (data.status === 'failed') {
          setIsTraining(false); setStatus('failed'); eventSource.close();
        }
      }
    };
  };

  const startTraining = async () => {
    setLogs(['[SYSTEM] Khởi tạo quá trình huấn luyện...']);
    setProgress(0);
    setIsTraining(true);
    setStatus('running');

    try {
      await axios.post('http://localhost:8001/start-training', {
        model_type: modelType,
        epochs: epochs
      });
      connectSSE();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs(prev => [...prev, `[ERROR] Không thể khởi động: ${msg}`]);
      setIsTraining(false); setStatus('failed');
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('http://localhost:8001/stop-training');
      setIsTraining(false);
      setStatus('idle');
    } catch (err) {
      console.error('Failed to stop training', err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 animate-in fade-in duration-500">
      {connectionError && (
        <div className="mb-8 p-4 bg-red-50 border-2 border-red-100 rounded-2xl text-red-600 font-black text-sm flex items-center gap-3">
           ⚠️ LỖI KẾT NỐI: Không thể liên lạc với máy chủ huấn luyện (Port 8001). Vui lòng chạy lại file run_all.bat!
        </div>
      )}
      <div className="flex flex-col lg:flex-row gap-8">
        
        {/* Left Column */}
        <div className="w-full lg:w-1/3 space-y-8">
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <h2 className="text-xl font-black text-slate-900 mb-6 flex items-center gap-2">
              ⚙️ Cấu Hình Huấn Luyện
            </h2>
            <div className="space-y-5">
              <div>
                <label className="block text-[10px] font-black text-slate-400 uppercase mb-2">Mô hình</label>
                <select 
                  value={modelType} onChange={(e) => setModelType(e.target.value)} disabled={isTraining}
                  className="w-full bg-slate-50 border-2 border-slate-100 rounded-xl px-4 py-3 text-sm font-bold"
                >
                  <option value="phobert">VinAI PhoBERT</option>
                  <option value="vibert">FPTAI ViBERT</option>
                  <option value="sbert">Vietnamese SBERT</option>
                  <option value="all">Tất cả mô hình ML</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] font-black text-slate-400 uppercase mb-2">Số Epochs</label>
                <input 
                  type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 1)} disabled={isTraining}
                  className="w-full bg-slate-50 border-2 border-slate-100 rounded-xl px-4 py-3 text-sm font-bold"
                />
              </div>
              
              {!isTraining ? (
                <button
                  onClick={startTraining}
                  className="w-full py-4 rounded-2xl font-black text-sm uppercase tracking-widest transition-all bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-100"
                >
                  ▶️ Bắt đầu huấn luyện
                </button>
              ) : (
                <button
                  onClick={stopTraining}
                  className="w-full py-4 rounded-2xl font-black text-sm uppercase tracking-widest transition-all bg-rose-500 text-white hover:bg-rose-600 shadow-lg shadow-rose-100"
                >
                  ⏹️ Dừng huấn luyện
                </button>
              )}
            </div>
          </div>

          <div className="bg-slate-900 rounded-3xl p-8 text-white">
            <h2 className="text-xl font-black mb-6">💻 Trạng Thái Hệ Thống</h2>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-[10px] font-black mb-2 text-slate-400 uppercase">
                  <span>CPU</span><span>{systemStats?.cpu}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: `${systemStats?.cpu || 0}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[10px] font-black mb-2 text-slate-400 uppercase">
                  <span>RAM</span><span>{systemStats?.ram_percent}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500" style={{ width: `${systemStats?.ram_percent || 0}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="flex-1 space-y-8">
          <div className="grid grid-cols-3 gap-4">
             <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center">
                <div className="text-[10px] font-black text-slate-400 uppercase mb-1">Trạng thái</div>
                <div className="text-sm font-black text-blue-600 uppercase">{status}</div>
             </div>
             <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center">
                <div className="text-[10px] font-black text-slate-400 uppercase mb-1">Mô hình</div>
                <div className="text-sm font-black text-purple-600 uppercase">{modelType}</div>
             </div>
             <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center relative overflow-hidden">
                <div className="text-[10px] font-black text-slate-400 uppercase mb-1">Tiến độ</div>
                <div className="text-sm font-black text-emerald-600 uppercase">{progress}%</div>
                {isTraining && details.total > 0 && (
                  <div className="absolute bottom-0 left-0 h-1 bg-emerald-100 w-full">
                    <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${progress}%` }}></div>
                  </div>
                )}
             </div>
          </div>

          {/* New Detailed Status Bar */}
          {isTraining && details.total > 0 && (
            <div className="bg-slate-800 rounded-2xl p-4 flex flex-wrap justify-between items-center gap-4 text-white border border-slate-700 shadow-lg animate-in slide-in-from-top-2">
               <div className="flex items-center gap-4">
                  <div>
                    <div className="text-[8px] font-black text-slate-500 uppercase">Tiến trình</div>
                    <div className="text-xs font-mono font-bold text-blue-400">{details.current} / {details.total} its</div>
                  </div>
                  <div className="h-8 w-px bg-slate-700"></div>
                  <div>
                    <div className="text-[8px] font-black text-slate-500 uppercase">Thời gian</div>
                    <div className="text-xs font-mono font-bold text-purple-400">{details.elapsed} &lt; {details.remaining}</div>
                  </div>
                  <div className="h-8 w-px bg-slate-700"></div>
                  <div>
                    <div className="text-[8px] font-black text-slate-500 uppercase">Tốc độ</div>
                    <div className="text-xs font-mono font-bold text-emerald-400">{details.speed}</div>
                  </div>
               </div>
               <div className="flex-1 max-w-xs bg-slate-900 h-2 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-1000" style={{ width: `${progress}%` }}></div>
               </div>
            </div>
          )}

          <div className="bg-slate-900 rounded-3xl shadow-2xl overflow-hidden flex flex-col border-4 border-slate-800">
            <div className="bg-slate-800 px-6 py-3 font-black text-[10px] text-slate-300 uppercase tracking-widest">
              Terminal Log Output
            </div>
            <div ref={terminalRef} className="h-[350px] p-6 font-mono text-[11px] text-blue-300 overflow-y-auto space-y-1 bg-[#020617]">
              {logs.map((log, i) => <div key={i}>{log}</div>)}
            </div>
            <div className="p-4 bg-slate-800 border-t border-slate-700">
               <div className="h-3 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-700">
                  <div className="h-full bg-blue-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
               </div>
            </div>
          </div>

          {/* Results Summary Chart (Manual) */}
          {Object.keys(results).length > 0 && (
            <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
              <h2 className="text-xl font-black text-slate-900 mb-8 uppercase tracking-tight">📊 Hiệu năng chi tiết</h2>
              <div className="space-y-10">
                {Object.entries(results).map(([name, data]) => {
                  const tn = data.confusion_matrix[0][0];
                  const fp = data.confusion_matrix[0][1];
                  const fn = data.confusion_matrix[1][0];
                  const tp = data.confusion_matrix[1][1];
                  
                  return (
                    <div key={name} className="border-b border-slate-50 pb-8 last:border-0">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-lg font-black text-blue-600 uppercase">{name}</span>
                        <span className="text-[10px] font-black text-slate-400 bg-slate-50 px-3 py-1 rounded-full">
                          THỜI GIAN: {data.duration.toFixed(1)}S
                        </span>
                      </div>
                      
                      {/* Main Progress Bars */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                        <div>
                          <div className="flex justify-between text-[10px] font-black mb-1 uppercase text-slate-500">
                            <span>Accuracy</span><span>{(data.accuracy * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                             <div className="h-full bg-blue-500" style={{ width: `${data.accuracy * 100}%` }}></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-[10px] font-black mb-1 uppercase text-slate-500">
                            <span>F1-Score</span><span>{(data.f1 * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                             <div className="h-full bg-purple-500" style={{ width: `${data.f1 * 100}%` }}></div>
                          </div>
                        </div>
                      </div>

                      {/* Detail Metrics Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                         <div className="bg-emerald-50 p-3 rounded-2xl border border-emerald-100 text-center">
                            <div className="text-[8px] font-black text-emerald-600 uppercase">True Positive (TP)</div>
                            <div className="text-xl font-black text-emerald-700">{tp}</div>
                         </div>
                         <div className="bg-emerald-50 p-3 rounded-2xl border border-emerald-100 text-center">
                            <div className="text-[8px] font-black text-emerald-600 uppercase">True Negative (TN)</div>
                            <div className="text-xl font-black text-emerald-700">{tn}</div>
                         </div>
                         <div className="bg-rose-50 p-3 rounded-2xl border border-rose-100 text-center">
                            <div className="text-[8px] font-black text-rose-600 uppercase">False Positive (FP)</div>
                            <div className="text-xl font-black text-rose-700">{fp}</div>
                         </div>
                         <div className="bg-rose-50 p-3 rounded-2xl border border-rose-100 text-center">
                            <div className="text-[8px] font-black text-rose-600 uppercase">False Negative (FN)</div>
                            <div className="text-xl font-black text-rose-700">{fn}</div>
                         </div>
                      </div>

                      <div className="mt-4 flex flex-wrap gap-4 text-[9px] font-black text-slate-400 uppercase">
                         <span>Precision: {(data.precision * 100).toFixed(1)}%</span>
                         <span>Recall: {(data.recall * 100).toFixed(1)}%</span>
                         <span>False Alarm: {(data.false_alarm_rate * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingPage;
