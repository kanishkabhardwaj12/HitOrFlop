import kagglehub

# Download latest version
path = kagglehub.dataset_download("theoverman/the-spotify-hit-predictor-dataset")

print("Path to dataset files:", path)
