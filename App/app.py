"""FastAPI server for CAD operations."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, TypedDict
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CadSeqProc.pipeline import CADPipeline, PipelineConfig
from CadSeqProc.enhanced_geometry.intelligent_cad import IntelligentCAD
from CadSeqProc.enhanced_geometry.llm_client import LLMClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="CADroid API",
    description="API for CAD model generation and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
config = PipelineConfig(
    model_type=os.getenv("DEFAULT_MODEL", "claude"),
    debug=os.getenv("DEBUG", "false").lower() == "true",
    cache_dir=os.getenv("CACHE_DIR", "./App/cache"),
    output_dir=os.getenv("OUTPUT_DIR", "./App/output")
)

# Initialize CAD system
llm_client = LLMClient(model_type=config.model_type)
cad_system = IntelligentCAD(llm_client)
pipeline = CADPipeline(config)

# Ensure output directories exist
os.makedirs(config.cache_dir, exist_ok=True)
os.makedirs(config.output_dir, exist_ok=True)

class AnalysisPattern(TypedDict):
    name: str
    description: str
    confidence: float

class MaterialSuggestion(TypedDict):
    material: str
    score: float

class ManufacturingAnalysis(TypedDict):
    best_process: str
    material_suggestions: List[MaterialSuggestion]
    analyses: Dict[str, Dict[str, Any]]

class Optimization(TypedDict):
    type: str
    suggestion: str

class PipelineResult(TypedDict):
    success: bool
    patterns: List[AnalysisPattern]
    manufacturing: ManufacturingAnalysis
    optimizations: List[Optimization]
    error: Optional[str]

class GenerateRequest(BaseModel):
    """Request model for CAD generation."""
    text: str
    process_type: str = "3d_printing_fdm"

class GenerateResponse(BaseModel):
    """Response model for CAD generation."""
    success: bool
    model_path: Optional[str] = None
    model_url: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalyzeRequest(BaseModel):
    """Request model for CAD analysis."""
    model_path: str
    process_type: str = "3d_printing_fdm"

class AnalyzeResponse(BaseModel):
    """Response model for CAD analysis."""
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "CADroid API",
        "version": "1.0.0",
        "description": "AI-powered CAD model generation and analysis",
        "endpoints": {
            "generate": "/generate - Generate CAD model from text description",
            "analyze": "/analyze - Analyze existing CAD model",
            "upload": "/upload - Upload CAD file for analysis",
            "download": "/download/{filename} - Download generated CAD file",
            "docs": "/docs - API documentation",
            "redoc": "/redoc - Alternative API documentation"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_cad(request: GenerateRequest) -> GenerateResponse:
    """Generate CAD model from text description."""
    try:
        # Process request through pipeline
        result = await pipeline.process(request.text)
        
        if not result["success"]:
            return GenerateResponse(
                success=False,
                error=result.get("error", "Unknown error occurred")
            )
        
        # Save results
        output_path = pipeline.save_results(result)
        
        # Format analysis results
        analysis = {
            "Patterns": [
                f"{p['name']}: {p['description']} (Confidence: {p['confidence']:.2f})"
                for p in result.get("patterns", [])
            ],
            "Manufacturing": {
                "Best Process": result.get("manufacturing", {}).get("best_process", ""),
                "Material Suggestions": [
                    f"{mat['material']}: {mat['score']:.1f}"
                    for mat in result.get("manufacturing", {}).get("material_suggestions", [])
                ],
                "Analysis": result.get("manufacturing", {}).get("analyses", {}).get(request.process_type, {})
            },
            "Optimizations": [
                f"{opt['type']}: {opt['suggestion']}"
                for opt in result.get("optimizations", [])
            ]
        }
        
        return GenerateResponse(
            success=True,
            model_path=str(output_path),
            model_url=f"/download/{output_path.name}",
            analysis=analysis
        )
        
    except Exception as e:
        return GenerateResponse(
            success=False,
            error=str(e)
        )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_cad(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze existing CAD model."""
    try:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        # Analyze model
        result = pipeline.analyze_model(model_path)
        
        return AnalyzeResponse(
            success=True,
            analysis=result
        )
        
    except Exception as e:
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload CAD file for analysis."""
    try:
        if file.filename is None:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        # Save uploaded file
        file_path = Path(config.cache_dir) / file.filename
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
            
        return JSONResponse({
            "success": True,
            "file_path": str(file_path)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@app.get("/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    """Download generated CAD file."""
    file_path = Path(config.output_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\nServer running at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"Alternative docs: http://{host}:{port}/redoc\n")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True
    )
