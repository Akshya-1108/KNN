# 📊 K-Nearest-Network Algorithm from scratch(Python)

This project demonstrates the **K-Nearest-Network** built from scratch using only NumPy, pandas and Matplotlib, with synthetic data generated via `scikit-learn`. It visually represents the clustering process and final centroids step-by-step.

---

## 🚀 Features

- Implements K-Means clustering without using `scikit-learn`'s built-in algorithm
- Step-by-step logic for:
  - Distance calculation
  - Building the KNN model
- Visualizes clustering results using Matplotlib
- Synthetic dataset generated using `make_blobs`
- Normalization of data for better convergence

---

## 🧠 How It Works

1. Generate a synthetic dataset with defined centers.
2. Normalize the dataset for balanced clustering.
3. Initialize `k` random centroids.
4. Iterate:
   - Assign each point to the nearest centroid
   - Update centroids to be the mean of assigned points
   - Repeat until convergence or max iterations
5. Visualize final clusters with colored points and centroid stars.

---

## 📦 Dependencies

- numpy
- matplotlib
- scikit-learn

Install them using:

````bash
pip install numpy matplotlib scikit-learn





📂 Usage

Clone the repository or download the Python file.

Run the script:

```bash
python KNN.py

(Optional) Uncomment the  print(dlist) andprint(labels,cnts) line for clear understanding of how the distance is calculated


📈 Output

- The script outputs a final scatter plot showing:
  - Clustered points in color
  - Centroids marked as black stars
````
