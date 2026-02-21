from .image import preprocess_image, extract_features

def run_pipeline(image_path: str):
    print("Preprocessing image...")
    preprocess_result = preprocess_image(image_path)
    print("Extracting features...")
    features = extract_features(
        preprocess_result["processed"],
        preprocess_result["edges"]
    )
    print("Extraction complete.")
    print(features)

if __name__ == "__main__":
    image_path = "../examples/example.jpg"
    run_pipeline(image_path)