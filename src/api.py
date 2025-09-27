"""
FastAPI Backend for Yoga Pose Detection Web Application
====================================================

This provides a RESTful API for the frontend to interact with the ML model.
Handles image upload, processing, and returns predictions.

Author: Shubham Prasad
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import traceback

from inference import YogaPoseDetector


# Initialize FastAPI app
app = FastAPI(
    title="Yoga Pose Detection API",
    description="AI-powered yoga pose classification using MediaPipe and PyTorch",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (serve the web interface)
app.mount("/static", StaticFiles(directory="../web"), name="static")

# Global detector instance
detector = None


def initialize_detector():
    """Initialize the pose detector on startup"""
    global detector
    try:
        detector = YogaPoseDetector(
            model_path="best_model.pth",
            scaler_path="scaler.pkl",
            classes_path="../data/pose_dataset_classes.json",
        )
        print("✅ Yoga pose detector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize detector when server starts"""
    success = initialize_detector()
    if not success:
        print("⚠️  Server starting without pose detection capability")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    try:
        with open("../web/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "message": "Yoga Pose Detection API is running",
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Pose detector not initialized")

    try:
        info = detector.get_model_info()
        return {"success": True, "model_info": info}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        )


@app.post("/detect-pose")
async def detect_pose(image: UploadFile = File(...)):
    """
    Detect and classify yoga pose from uploaded image

    Args:
        image: Uploaded image file

    Returns:
        JSON response with predictions and confidence scores
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Pose detector not initialized")

    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process the uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Run pose detection and classification
        result = detector.detect_pose_from_image(pil_image, draw_landmarks=True)

        if not result["success"]:
            return JSONResponse(
                status_code=200,
                content={"success": False, "error": result["error"], "predictions": []},
            )

        # Convert annotated image to base64 for frontend display
        annotated_image_b64 = None
        if result["annotated_image"] is not None:
            # Convert numpy array to PIL Image
            annotated_pil = Image.fromarray(result["annotated_image"])

            # Convert to base64
            buffer = io.BytesIO()
            annotated_pil.save(buffer, format="JPEG", quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            annotated_image_b64 = f"data:image/jpeg;base64,{img_str}"

        # Format response
        response = {
            "success": True,
            "predictions": result["predictions"],
            "top_prediction": {
                "pose_name": result["top_prediction"][0]
                if result["top_prediction"]
                else "Unknown",
                "confidence": result["confidence"],
            },
            "annotated_image": annotated_image_b64,
            "message": f"Detected: {result['top_prediction'][0] if result['top_prediction'] else 'Unknown pose'}",
        }

        return JSONResponse(content=response)

    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"Error processing image: {error_trace}")

        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect-pose-simple")
async def detect_pose_simple(image: UploadFile = File(...)):
    """
    Simplified pose detection endpoint (no annotated image)
    Faster response for mobile/low-bandwidth clients
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Pose detector not initialized")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process the uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Run pose detection without drawing landmarks (faster)
        result = detector.detect_pose_from_image(pil_image, draw_landmarks=False)

        if not result["success"]:
            return {"success": False, "error": result["error"], "predictions": []}

        return {
            "success": True,
            "predictions": result["predictions"],
            "top_prediction": {
                "pose_name": result["top_prediction"][0]
                if result["top_prediction"]
                else "Unknown",
                "confidence": result["confidence"],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/poses")
async def list_all_poses():
    """Get list of all supported yoga poses"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Pose detector not initialized")

    try:
        info = detector.get_model_info()

        # Get full list of class names if available
        if hasattr(detector, "class_names"):
            poses = [pose.replace("_", " ").title() for pose in detector.class_names]
        else:
            poses = []

        return {"success": True, "total_poses": len(poses), "poses": poses}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting pose list: {str(e)}"
        )


def main():
    """Run the FastAPI server"""
    print("🚀 Starting Yoga Pose Detection API Server")
    print("=" * 50)

    # Try to initialize detector before starting server
    if initialize_detector():
        print("✅ Model loaded successfully - API ready!")
    else:
        print("⚠️  Model not loaded - limited functionality")

    print("\n📡 Server will be available at:")
    print("   - Frontend: http://localhost:8000/")
    print("   - API docs: http://localhost:8000/docs")
    print("   - Health check: http://localhost:8000/health")

    # Start the server
    uvicorn.run(
        "api:app",  # module:app
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )


if __name__ == "__main__":
    main()
