"""
FastAPI server for face detection pipeline.
Provides REST API endpoints for face detection and analysis.
"""

from __future__ import annotations

import io
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from ..config import load_settings
from ..vision.detect import FaceDetector, FaceDet

# Initialize FastAPI app
app = FastAPI(
    title="Face Detection Pipeline API",
    description="REST API for face detection, tracking, and re-identification",
    version="0.1.0",
)

# Global detector instance (initialized on startup)
detector: Optional[FaceDetector] = None


class FaceDetectionResponse(BaseModel):
    """Response model for face detection"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    has_embedding: bool
    keypoints: Optional[List[List[float]]] = None


class DetectionResult(BaseModel):
    """Complete detection result"""
    success: bool
    num_faces: int
    faces: List[FaceDetectionResponse]
    message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the face detector on startup"""
    global detector
    try:
        settings = load_settings()
        detector = FaceDetector(
            det_size=settings.det_size,
            det_thresh=settings.det_thresh,
            max_faces=settings.max_faces,
            use_gpu=settings.use_gpu,
        )
        logger.info("Face detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize face detector: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Detection Pipeline API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
    }


@app.post("/detect", response_model=DetectionResult)
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Detection results with bounding boxes and metadata
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = detector.detect(image)
        
        # Convert to response format
        face_responses = []
        for face in faces:
            face_resp = FaceDetectionResponse(
                bbox=face.bbox_xyxy.tolist(),
                confidence=float(face.det_score),
                has_embedding=face.embedding is not None,
                keypoints=face.kps.tolist() if face.kps is not None else None,
            )
            face_responses.append(face_resp)
        
        return DetectionResult(
            success=True,
            num_faces=len(faces),
            faces=face_responses,
            message=f"Detected {len(faces)} face(s)",
        )
        
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/annotated")
async def detect_faces_annotated(file: UploadFile = File(...)):
    """
    Detect faces and return annotated image with bounding boxes.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Annotated image with face bounding boxes
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = detector.detect(image)
        
        # Draw bounding boxes
        for face in faces:
            x1, y1, x2, y2 = face.bbox_xyxy.astype(int).tolist()
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints if available
            if face.kps is not None:
                for (px, py) in face.kps.astype(int):
                    cv2.circle(image, (px, py), 2, (0, 255, 255), -1)
            
            # Add confidence label
            label = f"{face.det_score:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        return JSONResponse(
            content={
                "success": True,
                "num_faces": len(faces),
                "message": f"Detected {len(faces)} face(s)",
            },
            headers={
                "X-Num-Faces": str(len(faces)),
            },
        )
        
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
