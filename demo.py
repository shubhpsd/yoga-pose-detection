#!/usr/bin/env python3
"""
Quick demo script for yoga pose detection system
Run this to test the system with sample images
"""

import sys
import os
from pathlib import Path

# Get the current script directory
script_dir = Path(__file__).parent
src_dir = script_dir / "src"

# Add src directory to path
sys.path.insert(0, str(src_dir))

# Change to src directory so relative paths work
original_dir = Path.cwd()
os.chdir(src_dir)

try:
    from inference import YogaPoseDetector
except ImportError as e:
    print(
        "❌ Error importing modules. Make sure you have activated the virtual environment:"
    )
    print("   source .venv/bin/activate")
    print(f"   Error: {e}")
    os.chdir(original_dir)
    sys.exit(1)


def main():
    """Run a quick demo of the yoga pose detection system"""
    print("🧘 Yoga Pose Detection - Quick Demo")
    print("=" * 40)

    # Initialize detector
    try:
        detector = YogaPoseDetector()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        os.chdir(original_dir)
        return

    # Find sample images (go back to project root)
    samples_dir = Path("../samples")
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        os.chdir(original_dir)
        return

    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg"))

    if not image_files:
        print("❌ No sample images found in samples/ directory")
        print("💡 Add some .jpg files to the samples/ directory to test")
        return

    print(f"\n🔍 Testing with {len(image_files)} sample images:\n")

    # Test each image
    for i, image_path in enumerate(image_files[:3], 1):  # Test first 3 images
        print(f"📸 {i}. Testing: {image_path.name}")

        try:
            results = detector.detect_pose_from_image(str(image_path))

            if results["success"] and results["predictions"]:
                top_prediction = results["predictions"][0]
                pose_name, confidence = top_prediction
                print(f"   🎯 Detected: {pose_name} ({confidence:.1%} confidence)")

                # Show top 3 predictions
                print("   📊 Top 3 predictions:")
                for j, (pose, conf) in enumerate(results["predictions"][:3], 1):
                    print(f"      {j}. {pose} ({conf:.1%})")
            else:
                error_msg = results.get("error", "Unknown error")
                print(f"   ❌ {error_msg}")

        except Exception as e:
            print(f"   ❌ Error processing image: {e}")

        print()  # Empty line for readability

    print("🎉 Demo complete!")
    print("\n💻 To start the web interface:")
    print("   cd src")
    print("   python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000")
    print("   🌐 Then visit: http://localhost:8000")

    # Change back to original directory
    os.chdir(original_dir)


if __name__ == "__main__":
    main()
