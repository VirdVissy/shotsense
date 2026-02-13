"""
ShotSense API — FastAPI Backend.

REST API for the ShotSense photo curation system.

Endpoints:
    POST /api/projects                   — Create a new project
    POST /api/projects/{id}/references   — Upload reference photos, build prototype
    POST /api/projects/{id}/upload       — Upload candidate photos, get ranked results
    GET  /api/projects/{id}/analyze/{photo} — Detailed style analysis for one photo
    GET  /api/projects/{id}/styleguide   — Generate shareable style guide
    GET  /api/health                     — Health check
"""

import uuid
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from style_encoder import ShotSenseEncoder
from losses import compute_pseudo_labels
from explainability import StyleGradCAM


# ============================================================
# App Setup
# ============================================================

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder: Optional[ShotSenseEncoder] = None
gradcam: Optional[StyleGradCAM] = None
projects: Dict[str, dict] = {}

# Image preprocessing (CLIP normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

raw_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, gradcam
    print("Loading ShotSense model...")

    encoder = ShotSenseEncoder(pretrained=False, freeze_backbone=True).to(device)

    # Load trained weights if available
    checkpoint_path = Path("./checkpoints/checkpoint_best.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded trained weights")
    else:
        print("No checkpoint found — using untrained model (demo mode)")

    encoder.eval()
    gradcam = StyleGradCAM(encoder)
    print(f"Model loaded on {device}")
    yield


app = FastAPI(
    title="ShotSense API",
    description="AI-powered photo curation and style communication for creative teams",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    num_references: int
    has_prototype: bool

class PhotoScore(BaseModel):
    filename: str
    score: float
    attributes: Dict[str, float]

class StyleGuide(BaseModel):
    project_name: str
    prototype_attributes: Dict[str, float]
    reference_weights: List[float]
    description: str

class AnalysisResponse(BaseModel):
    score: float
    attributes: Dict[str, float]
    gaps: Dict[str, dict]
    heatmap_base64: str


# ============================================================
# Helper Functions
# ============================================================

def load_image_tensor(file_bytes: bytes) -> torch.Tensor:
    """Load an uploaded image into a preprocessed tensor."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def load_raw_tensor(file_bytes: bytes) -> torch.Tensor:
    """Load an uploaded image into a raw (unnormalized) tensor."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return raw_preprocess(img).unsqueeze(0).to(device)

def numpy_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy heatmap to a base64 PNG string."""
    arr_uint8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": encoder is not None,
        "device": str(device),
        "active_projects": len(projects),
    }


@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new curation project."""
    project_id = str(uuid.uuid4())
    projects[project_id] = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "created_at": datetime.now().isoformat(),
        "references": [],
        "reference_vectors": [],
        "prototype": None,
        "prototype_attributes": None,
        "reference_weights": [],
    }
    return ProjectResponse(
        id=project_id,
        name=project.name,
        description=project.description,
        created_at=projects[project_id]["created_at"],
        num_references=0,
        has_prototype=False,
    )


@app.post("/api/projects/{project_id}/references")
async def upload_references(
    project_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Upload reference photos and build a style prototype.
    
    These reference photos define the "target style" for the project.
    The system learns what visual style they share and creates a
    prototype that can be used to rank future uploads.
    """
    if project_id not in projects:
        raise HTTPException(404, "Project not found")
    if len(files) < 2:
        raise HTTPException(400, "Need at least 2 reference photos")

    project = projects[project_id]

    # Process reference images
    tensors = []
    raw_tensors = []
    filenames = []

    for file in files:
        file_bytes = await file.read()
        tensors.append(load_image_tensor(file_bytes))
        raw_tensors.append(load_raw_tensor(file_bytes))
        filenames.append(file.filename)

    # Stack into batch
    batch = torch.cat(tensors, dim=0)
    raw_batch = torch.cat(raw_tensors, dim=0)

    # Build prototype
    with torch.no_grad():
        prototype, weights = encoder.build_prototype(batch)

        # Get prototype attributes (average of reference attributes)
        result = encoder(batch)
        avg_attributes = {
            name: val.mean().item()
            for name, val in result["attributes"].items()
        }

    # Store in project
    project["references"] = filenames
    project["reference_vectors"] = result["style_vector"].cpu()
    project["prototype"] = prototype.cpu()
    project["prototype_attributes"] = avg_attributes
    project["reference_weights"] = weights.cpu().tolist()

    return {
        "message": f"Built style prototype from {len(files)} reference photos",
        "prototype_attributes": avg_attributes,
        "reference_weights": {
            name: round(w, 3)
            for name, w in zip(filenames, project["reference_weights"])
        },
    }


@app.post("/api/projects/{project_id}/upload", response_model=List[PhotoScore])
async def upload_and_rank(
    project_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Upload candidate photos and rank them by style match.
    
    Returns photos sorted by style similarity score (0-100).
    Higher = better match to the project's style prototype.
    """
    if project_id not in projects:
        raise HTTPException(404, "Project not found")

    project = projects[project_id]
    if project["prototype"] is None:
        raise HTTPException(400, "No style prototype — upload references first")

    prototype = project["prototype"].to(device)

    # Process and score each photo
    results = []
    for file in files:
        file_bytes = await file.read()
        tensor = load_image_tensor(file_bytes)

        with torch.no_grad():
            scores = encoder.score_against_prototype(tensor, prototype)

        results.append(PhotoScore(
            filename=file.filename,
            score=round(scores["scores"].item(), 1),
            attributes={
                name: round(val.item(), 3)
                for name, val in scores["attributes"].items()
            },
        ))

    # Sort by score (highest first)
    results.sort(key=lambda x: x.score, reverse=True)
    return results


@app.post("/api/projects/{project_id}/analyze")
async def analyze_photo(
    project_id: str,
    file: UploadFile = File(...),
):
    """
    Deep analysis of a single photo against the style prototype.
    
    Returns:
    - Style match score (0-100)
    - Attribute breakdown
    - Gap analysis (what's off and why)
    - Grad-CAM heatmap (which regions drive the style score)
    """
    if project_id not in projects:
        raise HTTPException(404, "Project not found")

    project = projects[project_id]
    if project["prototype"] is None:
        raise HTTPException(400, "No style prototype — upload references first")

    prototype = project["prototype"].to(device)
    file_bytes = await file.read()
    tensor = load_image_tensor(file_bytes)

    # Full analysis with Grad-CAM
    analysis = gradcam.generate_full_analysis(
        tensor,
        prototype,
        project["prototype_attributes"],
    )

    return AnalysisResponse(
        score=round(analysis["score"], 1),
        attributes=analysis["attributes"],
        gaps=analysis["gaps"],
        heatmap_base64=numpy_to_base64(analysis["heatmap"]),
    )


@app.get("/api/projects/{project_id}/styleguide")
async def get_style_guide(project_id: str):
    """
    Generate a shareable style guide for the project.
    
    Photographers can use this to understand the target style
    before a shoot — includes attribute targets and descriptions.
    """
    if project_id not in projects:
        raise HTTPException(404, "Project not found")

    project = projects[project_id]
    if project["prototype_attributes"] is None:
        raise HTTPException(400, "No style prototype — upload references first")

    attrs = project["prototype_attributes"]

    # Generate natural language description
    descriptions = []
    if attrs.get("warmth", 0.5) > 0.65:
        descriptions.append("warm, golden tones")
    elif attrs.get("warmth", 0.5) < 0.35:
        descriptions.append("cool, blue-toned")

    if attrs.get("contrast", 0.5) > 0.65:
        descriptions.append("high contrast")
    elif attrs.get("contrast", 0.5) < 0.35:
        descriptions.append("soft, low contrast")

    if attrs.get("saturation", 0.5) > 0.65:
        descriptions.append("vibrant, saturated colors")
    elif attrs.get("saturation", 0.5) < 0.35:
        descriptions.append("muted, desaturated palette")

    if attrs.get("brightness", 0.5) > 0.65:
        descriptions.append("bright, airy feel")
    elif attrs.get("brightness", 0.5) < 0.35:
        descriptions.append("dark, moody atmosphere")

    if attrs.get("grain", 0.5) > 0.5:
        descriptions.append("textured/grainy finish")
    else:
        descriptions.append("clean, smooth finish")

    description = f"Target style: {', '.join(descriptions)}." if descriptions else "Balanced, neutral style."

    return StyleGuide(
        project_name=project["name"],
        prototype_attributes=attrs,
        reference_weights=project["reference_weights"],
        description=description,
    )


@app.get("/api/projects")
async def list_projects():
    """List all active projects."""
    return [
        ProjectResponse(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            created_at=p["created_at"],
            num_references=len(p["references"]),
            has_prototype=p["prototype"] is not None,
        )
        for p in projects.values()
    ]


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
