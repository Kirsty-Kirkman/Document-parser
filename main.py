import os
from src import preprocess_image, extract_features

def run_pipeline(image_path: str, output_dir: str = "output"):
    print("Preprocessing image...")
    preprocess_result = preprocess_image(image_path, output_dir=output_dir)
    print("Extracting features...")
    features = extract_features(
        preprocess_result["processed"],
        preprocess_result["edges"],
        output_dir=output_dir
    )
    print("Extraction complete.")
    print(features)

if __name__ == "__main__":
    imgs = os.listdir("examples")
    for img in imgs:
        image_path = os.path.join("examples", img)
        output_dir = os.path.join("output", os.path.splitext(img)[0])
        run_pipeline(image_path, output_dir=output_dir)