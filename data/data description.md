# 🛰️ Data Description

This dataset consists of a hyperspectral image acquired on **May 27, 2020**, over the **Seville region in Spain**, using the **PRISMA satellite** ([PRISMA API](https://prisma.asi.it/)), in accordance with the *General Conditions for the Provision of PRISMA Products* and the *PRISMA License*.

The image has dimensions of **234 × 1215 × 1091**, where:

- **234** corresponds to the **spectral bands**,
- **1215 × 1091** is the **spatial resolution** (height × width),
- The original format was **GeoTIFF**, which we converted to **NumPy format (.npy)** for efficiency and ease of use.

In addition to the raw image, our team manually created ground truth masks with **six crop classes**:

| Class ID | Label                  |
|----------|------------------------|
| 0        | Background (not crop)  |
| 1        | Durum wheat            |
| 2        | Oranges                |
| 3        | Permanent grassland    |
| 4        | Rice                   |
| 5        | Sunflower              |
| 6        | Olives                 |

The ground truth is provided in 2 files:

- `train_crops.npy` – for training,
- `test_crops.npy` – for testing.

📄 More information about the PRISMA satellite can be found here: [http://prisma-i.it/](http://prisma-i.it/)
