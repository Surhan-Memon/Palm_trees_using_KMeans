# Image Color Quantization — KMeans Clustering

An unsupervised machine learning project that applies **KMeans clustering**
to compress an image by reducing it to a fixed number of representative colors,
a technique known as **color quantization**.

---

## Input

- **File:** `ck4cj8prb3n5l0706y00lz4ha.jpeg`
- **Resolution:** 2160 × 4096 pixels
- **Channels:** 3 (RGB)
- **Total pixels:** 8,847,360

---

## How It Works

Each pixel in an image has 3 values (R, G, B — each 0–255),
giving millions of possible colors. Color quantization reduces this
to just **K representative colors** by clustering similar pixels together.
Original image → reshape to 2D → KMeans clustering → replace each
pixel with its cluster center color → reshape back to image

---

## Process

### Step 1 — Load & Reshape
```python
image_as_array = mpimg.imread('image.jpeg')   # shape: (2160, 4096, 3)
image_as_array2d = image_as_array.reshape(h*w, c)  # shape: (8847360, 3)
```
Flattened from a 3D array (height × width × channels)
to a 2D array (pixels × RGB), so KMeans can treat each pixel as a data point.

### Step 2 — Cluster Pixels into K=6 Colors
```python
model = KMeans(n_clusters=6)
labels = model.fit_predict(image_as_array2d)
```
Every pixel is assigned to one of **6 clusters** based on color similarity.

### Step 3 — Extract Cluster Center Colors
```python
rgb_codes = model.cluster_centers_.round(0).astype(int)
```

The 6 representative colors found:

| Cluster | R | G | B | Approximate Color |
|---------|---|---|---|-------------------|
| 0 | 63 | 62 | 59 | Dark gray |
| 1 | 198 | 196 | 189 | Light gray/off-white |
| 2 | 167 | 163 | 154 | Medium gray |
| 3 | 97 | 96 | 92 | Mid-dark gray |
| 4 | 30 | 32 | 27 | Near black |
| 5 | 127 | 129 | 127 | Middle gray |

> The palette is entirely grayscale tones — indicating the source image
> has very low color saturation (likely a black & white or monochrome photo).

### Step 4 — Reconstruct Quantized Image
```python
quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
plt.imshow(quantized_image)
```
Each pixel replaced with its cluster's representative RGB color,
then reshaped back to the original image dimensions.

---

## Libraries Used

- `numpy` — array operations and reshaping
- `matplotlib` — image loading (`mpimg`) and display (`imshow`)
- `scikit-learn`:
  - `KMeans` — pixel color clustering

---


## Key Takeaways

| Parameter | Value |
|-----------|-------|
| Image size | 2160 × 4096 px |
| Total pixels clustered | 8,847,360 |
| Number of colors (K) | 6 |
| Color palette | 6 grayscale tones |

> **Lesson 1:** Any image can be treated as a dataset of pixels —
> each pixel is simply a point in 3D RGB space.

> **Lesson 2:** Increasing K preserves more detail;
> decreasing K creates a more abstract, posterized look.
> Try K=2 for silhouette effects or K=16+ for near-original quality.

> **Lesson 3:** This is a practical demonstration of how KMeans
> works on real high-dimensional data — 8.8 million data points
> clustered in seconds.
