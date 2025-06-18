"""
DeepSeek Local Model Server
A FastAPI-based server for running DeepSeek models locally
"""

import os
import logging
import sys
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        pipeline
    )
    import uvicorn
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you've activated the virtual environment and installed requirements:")
    print("   source venv/bin/activate  # Linux/Mac")
    print("   venv\\Scripts\\activate.bat  # Windows")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
text_generator = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelManager:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        
        logger.info(f"🔧 Initializing ModelManager")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🗜️ Quantization: {self.use_quantization}")
        logger.info(f"🤖 Model: {self.model_name}")
        
    def load_model(self):
        """Load the DeepSeek model and tokenizer"""
        global model, tokenizer, text_generator
        
        try:
            logger.info(f"📥 Loading model: {self.model_name}")
            
            # Load tokenizer first
            logger.info("📝 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure model loading based on device
            logger.info("🧠 Loading model...")
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # CPU-specific configuration
            if self.device == "cpu":
                logger.info("💻 Configuring for CPU-only mode...")
                model_kwargs.update({
                    "torch_dtype": torch.float32,  # Use float32 for CPU
                    "device_map": None,  # Don't use device_map for CPU
                })
                
                # Don't use quantization on CPU
                if self.use_quantization:
                    logger.warning("⚠️ Quantization disabled for CPU mode")
                    self.use_quantization = False
                    
            else:
                # GPU configuration
                logger.info("🎮 Configuring for GPU mode...")
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                })
                
                # Configure quantization for GPU
                if self.use_quantization:
                    logger.info("🗜️ Setting up 4-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to device if CPU
            if self.device == "cpu":
                model = model.to("cpu")
                logger.info("📱 Model moved to CPU")
            
            # Create text generation pipeline
            logger.info("🔧 Creating text generation pipeline...")
            
            pipeline_kwargs = {
                "task": "text-generation",
                "model": model,
                "tokenizer": tokenizer,
                "return_full_text": False,
            }
            
            # Don't set device for CPU
            if self.device != "cpu":
                pipeline_kwargs["device_map"] = "auto"
            
            text_generator = pipeline(**pipeline_kwargs)
            
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise e

    def unload_model(self):
        """Unload the model to free memory"""
        global model, tokenizer, text_generator
        
        logger.info("🧹 Unloading model...")
        
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        if text_generator is not None:
            del text_generator
            text_generator = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Model unloaded successfully!")

# Initialize model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting DeepSeek Model Server...")
    try:
        model_manager.load_model()
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Continue anyway for health checks
    yield
    # Shutdown
    logger.info("🛑 Shutting down DeepSeek Model Server...")
    model_manager.unload_model()

# Create FastAPI app
app = FastAPI(
    title="DeepSeek Local Server",
    description="A local server for running DeepSeek models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🤖 DeepSeek Local Server is running!",
        "model": model_manager.model_name,
        "device": model_manager.device,
        "quantization": model_manager.use_quantization,
        "status": "ready" if model is not None else "loading"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved(),
            "gpu_count": torch.cuda.device_count()
        }
    
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "device": model_manager.device,
        "cuda_available": torch.cuda.is_available(),
        "model_name": model_manager.model_name,
        "quantization_enabled": model_manager.use_quantization,
        **memory_info
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    global text_generator
    
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    try:
        # Format messages for the model
        conversation = ""
        for message in request.messages:
            if message.role == "system":
                conversation += f"System: {message.content}\n"
            elif message.role == "user":
                conversation += f"User: {message.content}\n"
            elif message.role == "assistant":
                conversation += f"Assistant: {message.content}\n"
        
        conversation += "Assistant:"
        
        # Generate response
        outputs = text_generator(
            conversation,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]['generated_text'].strip()
        
        # Create response
        response = ChatResponse(
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(tokenizer.encode(conversation)),
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": len(tokenizer.encode(conversation)) + len(tokenizer.encode(generated_text))
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    """Simple text generation endpoint"""
    global text_generator
    
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    try:
        outputs = text_generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return {
            "prompt": prompt,
            "generated_text": outputs[0]['generated_text'],
            "model": model_manager.model_name
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🏃‍♂️ Starting server directly...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )