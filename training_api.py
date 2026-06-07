import os
import sys
import json
import time
import psutil
import torch
import multiprocessing
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import training functions
from src.train import train_all_models
from src.train_phobert import train_phobert
from src.train_sbert import train_sbert
from src.train_vibert import train_vibert

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for training
training_status = {
    "is_training": False,
    "current_model": None,
    "progress": 0,
    "start_time": None,
    "estimated_end_time": None,
    "logs": []
}

# Queue for inter-process communication
log_queue = multiprocessing.Queue()

class TrainingRequest(BaseModel):
    model_type: str  # 'all', 'phobert', 'sbert', 'vibert'
    epochs: Optional[int] = 10

class CustomStream:
    def __init__(self, queue, original_stream):
        self.queue = queue
        self.original_stream = original_stream

    def write(self, data):
        if data.strip():
            self.queue.put(json.dumps({"type": "log", "content": data.strip()}))
        self.original_stream.write(data)

    def flush(self):
        self.original_stream.flush()

def training_process_wrapper(model_type, epochs, data_dir, models_dir, queue):
    # Redirect stdout to capture logs
    sys.stdout = CustomStream(queue, sys.stdout)
    
    try:
        if model_type == 'all':
            train_all_models(data_dir, models_dir)
        elif model_type == 'phobert':
            # Note: The original script has hardcoded epochs, 
            # we might need to modify it or accept it's fixed for now.
            # To minimize changes to original code, we'll just call it.
            # Ideally, we should pass epochs to it.
            train_phobert(data_dir, models_dir)
        elif model_type == 'sbert':
            train_sbert(data_dir, models_dir)
        elif model_type == 'vibert':
            train_vibert(data_dir, models_dir)
        
        queue.put(json.dumps({"type": "status", "status": "completed"}))
    except Exception as e:
        queue.put(json.dumps({"type": "status", "status": "failed", "error": str(e)}))

@app.post("/start-training")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    global training_status
    if training_status["is_training"]:
        return {"error": "Training already in progress"}

    training_status["is_training"] = True
    training_status["current_model"] = req.model_type
    training_status["start_time"] = time.time()
    training_status["logs"] = []
    
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    
    # Use multiprocessing for training
    p = multiprocessing.Process(
        target=training_process_wrapper, 
        args=(req.model_type, req.epochs, data_dir, models_dir, log_queue)
    )
    p.start()
    
    return {"message": f"Started training for {req.model_type}"}

@app.get("/training-events")
async def training_events():
    def event_generator():
        global training_status
        while True:
            try:
                # Try to get data from queue with a timeout
                data = log_queue.get(timeout=1.0)
                parsed = json.loads(data)
                
                if parsed["type"] == "status":
                    if parsed["status"] in ["completed", "failed"]:
                        training_status["is_training"] = False
                
                yield f"data: {data}\n\n"
            except multiprocessing.queues.Empty:
                if not training_status["is_training"]:
                    break
                yield ": keep-alive\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/system-stats")
async def get_system_stats():
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    gpu_stats = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_stats.append({
                "name": props.name,
                "usage": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
                "memory_total": props.total_memory / (1024**2),
                "memory_used": torch.cuda.memory_allocated(i) / (1024**2)
            })
    
    return {
        "cpu": cpu_usage,
        "ram_total": ram.total / (1024**3),
        "ram_used": ram.used / (1024**3),
        "ram_percent": ram.percent,
        "gpus": gpu_stats,
        "disk": psutil.disk_usage('/').percent
    }

@app.get("/training-results")
async def get_training_results():
    metrics_path = os.path.join(BASE_DIR, "models", "models_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
