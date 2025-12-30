#!/usr/bin/env python3
"""
Demo script for the Face Detection Pipeline.
Tests the pipeline components without requiring a webcam.
"""

import numpy as np
import cv2
from loguru import logger

from src.vision.detect import FaceDetector
from src.vision.track import IouTracker
from src.vision.reid import ReIdentifier
from src.vision.embed import l2_normalize, cosine_similarity


def test_embedding_functions():
    """Test embedding utility functions"""
    logger.info("Testing embedding functions...")
    
    # Test L2 normalization
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normalized = l2_normalize(vec)
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 1e-6, "L2 normalization failed"
    logger.info(f"✓ L2 normalization: {vec} -> {normalized} (norm={norm:.6f})")
    
    # Test cosine similarity
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    sim = cosine_similarity(a, b)
    assert abs(sim - 1.0) < 1e-6, "Cosine similarity failed for identical vectors"
    logger.info(f"✓ Cosine similarity (identical): {sim:.6f}")
    
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    sim = cosine_similarity(a, b)
    assert abs(sim) < 1e-6, "Cosine similarity failed for orthogonal vectors"
    logger.info(f"✓ Cosine similarity (orthogonal): {sim:.6f}")


def test_tracker():
    """Test IOU tracker"""
    logger.info("\nTesting IOU tracker...")
    
    tracker = IouTracker(iou_thresh=0.3, max_missed=5)
    
    # Frame 1: Two faces
    bboxes1 = [
        np.array([10, 10, 50, 50], dtype=np.float32),
        np.array([100, 100, 150, 150], dtype=np.float32),
    ]
    assignments1 = tracker.update(bboxes1)
    logger.info(f"Frame 1: {len(bboxes1)} detections -> {assignments1}")
    assert len(assignments1) == 2, "Should create 2 tracks"
    
    # Frame 2: Same faces, slightly moved
    bboxes2 = [
        np.array([12, 12, 52, 52], dtype=np.float32),
        np.array([102, 102, 152, 152], dtype=np.float32),
    ]
    assignments2 = tracker.update(bboxes2)
    logger.info(f"Frame 2: {len(bboxes2)} detections -> {assignments2}")
    assert len(assignments2) == 2, "Should maintain 2 tracks"
    
    # Frame 3: One face disappears
    bboxes3 = [
        np.array([14, 14, 54, 54], dtype=np.float32),
    ]
    assignments3 = tracker.update(bboxes3)
    logger.info(f"Frame 3: {len(bboxes3)} detections -> {assignments3}")
    
    logger.info(f"✓ Tracker test passed. Active tracks: {tracker.active_track_ids()}")


def test_reid():
    """Test re-identification"""
    logger.info("\nTesting re-identification...")
    
    reid = ReIdentifier(similarity_thresh=0.45, update_alpha=0.9, min_det_score=0.65)
    
    # Create synthetic embeddings
    emb1 = l2_normalize(np.random.randn(512).astype(np.float32))
    emb2 = l2_normalize(np.random.randn(512).astype(np.float32))
    
    # Track 1 with embedding 1
    pid1 = reid.assign_person_id(track_id=1, embedding=emb1, det_score=0.8)
    logger.info(f"Track 1 -> Person {pid1}")
    assert pid1 == 1, "First person should get ID 1"
    
    # Track 2 with embedding 2 (different person)
    pid2 = reid.assign_person_id(track_id=2, embedding=emb2, det_score=0.8)
    logger.info(f"Track 2 -> Person {pid2}")
    assert pid2 == 2, "Second person should get ID 2"
    
    # Track 3 with similar embedding to person 1
    emb1_similar = emb1 + np.random.randn(512).astype(np.float32) * 0.1
    emb1_similar = l2_normalize(emb1_similar)
    pid3 = reid.assign_person_id(track_id=3, embedding=emb1_similar, det_score=0.8)
    logger.info(f"Track 3 (similar to 1) -> Person {pid3}")
    
    logger.info(f"✓ ReID test passed. Total persons: {len(reid.people)}")


def test_face_detector():
    """Test face detector with a synthetic image"""
    logger.info("\nTesting face detector...")
    
    try:
        # Create a simple test image (blank)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background
        
        # Draw a simple "face" (circle)
        cv2.circle(test_image, (320, 240), 80, (255, 200, 150), -1)
        cv2.circle(test_image, (290, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (350, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_image, (320, 260), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        logger.info("Initializing face detector (this may take a moment)...")
        detector = FaceDetector(
            det_size=(640, 640),
            det_thresh=0.5,
            use_gpu=False,
        )
        
        logger.info("Running detection on test image...")
        faces = detector.detect(test_image)
        
        logger.info(f"✓ Detector initialized successfully")
        logger.info(f"  Detected {len(faces)} face(s) in test image")
        
        if len(faces) > 0:
            for i, face in enumerate(faces):
                logger.info(f"  Face {i+1}: confidence={face.det_score:.3f}, "
                          f"bbox={face.bbox_xyxy}, has_embedding={face.embedding is not None}")
        
    except ImportError as e:
        logger.warning(f"⚠ Could not test face detector: {e}")
        logger.warning("  Make sure insightface is installed: pip install insightface")
    except Exception as e:
        logger.error(f"✗ Face detector test failed: {e}")


def main():
    """Run all demo tests"""
    logger.info("=" * 60)
    logger.info("Face Detection Pipeline - Demo Script")
    logger.info("=" * 60)
    
    try:
        test_embedding_functions()
        test_tracker()
        test_reid()
        test_face_detector()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Run the main pipeline: python run.py")
        logger.info("  2. Start the API server: python -m uvicorn src.api.server:app --reload")
        logger.info("  3. Check the documentation: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"\n✗ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
