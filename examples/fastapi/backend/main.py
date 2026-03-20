import os
import sys
import json
import logging
import asyncio
import psutil
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# --- Environment Configuration ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

if "PYTHONPATH" in os.environ:
    paths = os.environ["PYTHONPATH"].split(os.pathsep)
    for p in reversed(paths):
        if p and p not in sys.path:
            sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Native NLP Engine Integration ---
try:
    from nlp_engine import AsyncNLPEngine, FractalAddon, DeduplicationAddon
except ImportError as e:
    print(f"DEBUG: sys.path is {sys.path}")
    print(f"DEBUG: PYTHONPATH is {os.environ.get('PYTHONPATH')}")
    raise ImportError(
        f"Could not find the 'nlp_engine' native module. Error: {e}\n"
        "Please build it using './dev.sh --python' before starting the server."
    ) from e

app = FastAPI(title="NLP Engine API", version="2.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

engine = AsyncNLPEngine()
active_tasks_tracker = {}

class ProcessingRequest(BaseModel):
    text: str
    plugin: str = "default"
    options: Dict[str, Any] = {}
    streaming: bool = False
    session_id: Optional[str] = None

class ProcessingResponse(BaseModel):
    result: str
    task_id: Optional[str] = None
    status: str = "completed"

def refresh_markov_models():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    models_dir = os.path.join(data_path, "models")
    app.state.available_models = []

    if os.path.exists(models_dir):
        for model_file in sorted(os.listdir(models_dir)):
            if model_file.endswith(".json"):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(models_dir, model_file)
                engine.load_markov_model(model_path, model_name)
                app.state.available_models.append(model_name)
                logger.info(f"Registered Markov model: {model_name}")

    # Re-sync Fractal dependencies to ensure it picks up the newly loaded Markov sources
    try:
        addons = engine.get_all_addons()
        if "fractal_generator" in addons:
            fractal = addons["fractal_generator"]
            # Re-run registration logic to link the primary markov_generator
            # If a specific model is selected in UI, it should be the source
            engine.register_fractal_addon(fractal)
            logger.info("Fractal dependencies re-synced after model refresh.")
    except Exception as e:
        logger.warning(f"Failed to re-sync fractal dependencies: {e}")

@app.on_event("startup")
async def startup_event():
    try:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        if not engine.load_model(data_path):
            logger.error(f"Failed to load model resources from: {data_path}")
        engine.initialize()
        logger.info(f"Native NLP Engine initialized with resources from {data_path}")

        # Register Addons BEFORE refreshing models so dependencies can be linked
        try:
            fractal = FractalAddon()
            engine.register_fractal_addon(fractal)
            logger.info("Fractal Text Generator addon registered.")
        except Exception as fe:
            logger.warning(f"Could not initialize Fractal addon: {fe}")

        try:
            dedupe = DeduplicationAddon()
            engine.register_dedupe_addon(dedupe)
            logger.info("Deduplication addon registered.")
        except Exception as de:
            logger.warning(f"Could not initialize Deduplication addon: {de}")

        refresh_markov_models()

    except Exception as e:
        logger.error(f"Failed to initialize native engine: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if engine:
        engine.shutdown()
        logger.info("Native NLP Engine shut down")

class TrainingRequest(BaseModel):
    category: str
    text: Optional[str] = None
    ngram_size: int = 2

@app.post("/detect-language")
async def detect_language(request: ProcessingRequest):
    try:
        res = engine.process_text_sync(request.text, "language", request.options)
        return json.loads(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spell-check")
async def spell_check(request: ProcessingRequest):
    try:
        res = engine.process_text_sync(request.text, "spell_check", request.options)
        return json.loads(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment")
async def analyze_sentiment(request: ProcessingRequest):
    try:
        res = engine.process_sync(request.text, "sentiment", request.options, request.session_id or "")
        return json.loads(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: ProcessingRequest):
    internal_task_id = f"gen_{int(time.time() * 1000)}"
    active_tasks_tracker[internal_task_id] = {"type": "MarkovGen", "start": time.time()}
    try:
        # Map plugin to model name if provided
        method = request.plugin if (request.plugin and request.plugin != "default") else "markov_generator"

        # If calling fractal_generator, ensure the source model is linked
        if method == "fractal_generator":
            # Default to generic_novel if no model specified
            source_model = request.options.get("model") or "generic_novel"
            addons = engine.get_all_addons()
            if "fractal_generator" in addons:
                fractal = addons["fractal_generator"]
                # This call internally links the source model to the fractal engine
                engine.register_fractal_addon(fractal, source_model)
                logger.info(f"Linked Fractal engine to source model: {source_model}")

        if method == "markov_generator" and request.options.get("model"):
            method = request.options["model"]

        safe_options = {k: str(v) for k, v in request.options.items()}
        res = engine.process_sync(request.text, method, safe_options, request.session_id or "")
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            return {"output": res, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_tasks_tracker.pop(internal_task_id, None)
        logger.info(f"Task complete: {internal_task_id}")

@app.get("/generate-stream")
async def generate_text_stream(
    seed: str,
    model: str = "generic_novel",
    length: int = 150,
    top_p: float = 0.9,
    temperature: float = 1.0,
    n_gram: int = 2,
    use_hybrid: str = "false",
    semantic_filter: float = 0.3,
    session_id: Optional[str] = None
):
    internal_task_id = f"gen_stream_{int(time.time() * 1000)}"
    active_tasks_tracker[internal_task_id] = {"type": "MarkovStream", "start": time.time()}

    options = {
        "length": str(length),
        "top_p": str(top_p),
        "temperature": str(temperature),
        "n_gram": str(n_gram),
        "use_hybrid": use_hybrid,
        "semantic_filter": str(semantic_filter)
    }

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def cpp_callback(chunk: str, is_final: bool):
        loop.call_soon_threadsafe(queue.put_nowait, {"chunk": chunk, "is_final": is_final})

    async def event_generator():
        try:
            engine.stream_text(seed, model, cpp_callback, options, session_id or "")
            logger.info(f"Engine stream initiated for model: {model} with seed: {seed}")
            while True:
                data = await queue.get()
                logger.info(f"Yielding data: {data}")
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("is_final"):
                    break
        except Exception as e:
            logger.error(f"Error in Markov stream: {e}")
            yield f"data: {json.dumps({'chunk': '', 'is_final': True, 'error': str(e)})}\n\n"
        finally:
            active_tasks_tracker.pop(internal_task_id, None)
            logger.info(f"Task complete: {internal_task_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/train-model")
async def train_model(request: TrainingRequest):
    internal_task_id = f"train_{int(time.time() * 1000)}"
    active_tasks_tracker[internal_task_id] = {"type": "ModelTraining", "start": time.time()}
    try:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        source_dir = os.path.join(data_path, "training")
        models_dir = os.path.join(data_path, "models")
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        category_name = request.category.strip().lower()
        if category_name in ["generic_novel", "default", "language", "spell_check"]:
            category_name = f"user_{category_name}"
        if not category_name.startswith("_") and not category_name.startswith("user_"):
            category_name = f"_{category_name}"
        source_path = os.path.join(source_dir, f"{category_name}_source.txt")
        model_path = os.path.join(models_dir, f"{category_name}.json")
        if request.text:
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(request.text)
        if not os.path.exists(source_path):
            raise HTTPException(status_code=400, detail=f"Source file {source_path} not found.")
        success = engine.train_markov_model(source_path, model_path, request.ngram_size)
        if success:
            refresh_markov_models()
            return {"status": "success", "model": category_name, "ngram_size": request.ngram_size}
        else:
            raise HTTPException(status_code=500, detail="Native training failed.")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_tasks_tracker.pop(internal_task_id, None)
        logger.info(f"Task complete: {internal_task_id}")

@app.post("/semantic")
async def semantic_analysis(request: ProcessingRequest):
    internal_task_id = f"vec_{int(time.time() * 1000)}"
    active_tasks_tracker[internal_task_id] = {"type": "VectorEngine", "start": time.time()}
    try:
        method = request.plugin if request.plugin != "default" else "vector_engine"
        safe_options = {k: str(v) for k, v in request.options.items()}
        res = engine.process_sync(request.text, method, safe_options, request.session_id or "")
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            return {"output": res, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_tasks_tracker.pop(internal_task_id, None)
        logger.info(f"Task complete: {internal_task_id}")

@app.post("/async-process")
async def async_process_text(request: ProcessingRequest):
    try:
        task_id = engine.process_text_async(request.text, request.plugin, request.options, request.session_id or "")
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Async processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/{task_id}")
async def stream_results(
    task_id: str,
    text: str,
    pos_tagging: Optional[str] = None,
    terminology: Optional[str] = None,
    safety: Optional[str] = None,
    session_id: Optional[str] = None
):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    options = {}
    if pos_tagging: options["pos_tagging"] = pos_tagging
    if terminology: options["terminology"] = terminology
    if safety: options["safety"] = safety
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    def cpp_callback(chunk: str, is_final: bool):
        loop.call_soon_threadsafe(queue.put_nowait, {"chunk": chunk, "is_final": is_final})
    async def event_generator():
        internal_task_id = f"stream_{int(time.time() * 1000)}"
        active_tasks_tracker[internal_task_id] = {"type": "LinguisticStream", "start": time.time()}
        try:
            engine.stream_text(text, "default", cpp_callback, options, session_id or "")
            while True:
                data = await queue.get()
                logger.info(f"Yielding data: {data}")
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("is_final"):
                    break
        except Exception as e:
            logger.error(f"Error in SSE: {e}")
            yield f"data: {json.dumps({'chunk': '', 'is_final': True, 'error': str(e)})}\n\n"
        finally:
            active_tasks_tracker.pop(internal_task_id, None)
            logger.info(f"Task complete: {internal_task_id}")
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage_mb = mem_info.rss / (1024 * 1024)
    cpu_usage = process.cpu_percent(interval=None)
    now = time.time()
    active_tasks = [
        {"id": tid, "type": tdata["type"], "elapsed": round(now - tdata["start"], 1)}
        for tid, tdata in active_tasks_tracker.items()
    ]
    return {
        "status": "healthy",
        "engine_type": "native",
        "engine_ready": engine.is_ready() if engine else False,
        "available_models": getattr(app.state, "available_models", []),
        "stats" : {
            "ram_mb": round(ram_usage_mb, 2),
            "cpu_percent": cpu_usage,
            "uptime_seconds": int(now - process.create_time()),
            "threads": process.num_threads(),
            "active_tasks": active_tasks
        }
    }

CLIENT_PATH = os.path.join(os.path.dirname(__file__), "..", "client", "dist")
if os.path.exists(CLIENT_PATH):
    app.mount("/static", StaticFiles(directory=CLIENT_PATH), name="static")
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        if not full_path:
            return FileResponse(os.path.join(CLIENT_PATH, "index.html"))
        file_path = os.path.join(CLIENT_PATH, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(CLIENT_PATH, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
