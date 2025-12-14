# GEMM
`experimental C++26`, `SIMD`, `matrix multiplication`

> Another low latency experiment ported over because its deserves its own repository

## Motivation

Working with [SWAR](https://github.com/lim-james/swar-stoi) wasn't enough, I
wanted to experience the true SIMD benefits. Was an opportunity to be exposed to
cpp26 documentations and experimental features.

# Benchmark Results

The following tables present the performance metrics for different algorithms across various problem sizes.

**Legend:**
* **k, M, G:** Kilo ($10^3$), Mega ($10^6$), Giga ($10^9$).
* **Scaling (x):** The value in parentheses compares the algorithm's performance to the **Naive** implementation.
    * For **GOps** and **Bandwidth**, this is **Throughput Improvement** (Algo / Naive). Values $> 1.0$x indicate higher throughput.

## GOps (Billions of Operations per Second)

|   Size | Naive   | Transposed      | Simd            | Tiling          |
|-------:|:--------|:----------------|:----------------|:----------------|
|      4 | 183.40  | 182.12 (0.99x)  | 182.93 (1.00x)  | 150.27 (0.82x)  |
|      8 | 1.49k   | 1.46k (0.98x)   | 1.47k (0.99x)   | 1.47k (0.99x)   |
|     16 | 11.12k  | 11.94k (1.07x)  | 11.56k (1.04x)  | 10.73k (0.96x)  |
|     32 | 96.97k  | 92.94k (0.96x)  | 92.89k (0.96x)  | 93.50k (0.96x)  |
|     64 | 739.54k | 749.09k (1.01x) | 1.04M (1.41x)   | 753.08k (1.02x) |
|    128 | 5.97M   | 5.98M (1.00x)   | 5.54M (0.93x)   | 6.03M (1.01x)   |
|    256 | 47.21M  | 48.61M (1.03x)  | 48.14M (1.02x)  | 48.01M (1.02x)  |
|    512 | 449.29M | 396.31M (0.88x) | 375.03M (0.83x) | 381.22M (0.85x) |
|   1024 | 274.70M | 3.64G (13.26x)  | 3.74G (13.61x)  | 3.14G (11.44x)  |
|   2048 | 202.71M | 3.01G (14.83x)  | 4.45G (21.93x)  | 7.36G (36.33x)  |
|   4096 | 205.20M | 3.40G (16.58x)  | 5.62G (27.39x)  | 8.98G (43.75x)  |

## Bandwidth

|   Size | Naive   | Transposed      | Simd            | Tiling          |
|-------:|:--------|:----------------|:----------------|:----------------|
|      4 | 275.11  | 273.19 (0.99x)  | 274.40 (1.00x)  | 225.40 (0.82x)  |
|      8 | 1.12k   | 1.10k (0.98x)   | 1.10k (0.99x)   | 1.10k (0.99x)   |
|     16 | 4.17k   | 4.48k (1.07x)   | 4.33k (1.04x)   | 4.02k (0.96x)   |
|     32 | 18.18k  | 17.43k (0.96x)  | 17.42k (0.96x)  | 17.53k (0.96x)  |
|     64 | 69.33k  | 70.23k (1.01x)  | 97.86k (1.41x)  | 70.60k (1.02x)  |
|    128 | 279.80k | 280.52k (1.00x) | 259.88k (0.93x) | 282.78k (1.01x) |
|    256 | 1.11M   | 1.14M (1.03x)   | 1.13M (1.02x)   | 1.13M (1.02x)   |
|    512 | 5.27M   | 4.64M (0.88x)   | 4.39M (0.83x)   | 4.47M (0.85x)   |
|   1024 | 1.61M   | 21.35M (13.26x) | 21.90M (13.61x) | 18.41M (11.44x) |
|   2048 | 593.86k | 8.81M (14.83x)  | 13.02M (21.93x) | 21.58M (36.33x) |
|   4096 | 300.59k | 4.98M (16.58x)  | 8.23M (27.39x)  | 13.15M (43.75x) |
