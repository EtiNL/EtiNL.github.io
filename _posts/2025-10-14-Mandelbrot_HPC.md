---
layout: post
title: "HPC Mandelbrot set: MPI & CUDA"
date: 2025-10-14
math: true
tags: [HPC, MPI, CUDA, Mandelbrot, GPU, Parallelism]
toc: true
---
This post summarizes a project completed for my M2 High Performance Computing class.

<p align="center">
  <img src="/assets/img/HPC_mandelbrot/mandel.png" alt="Mandelbrot set — cover" width="70%">
</p>

[Download the full report (PDF)]({{ "/assets/pdf/report_mandelbrot_project.pdf" | relative_url }})

## TLDR
- Sequential baseline (1 process) renders a $1000\times1000$ image in $\approx 2.31\,\text{s}$.
- Static 1D MPI decomposition hits load imbalance near the set boundary and stalls at $\approx 2.34\times$ with 8 processes.
- Master–Worker MPI removes the imbalance and reaches $\approx 6.70\times$ speedup on 9 processes (8 workers), with efficiency $\approx (p-1)/p$.
- A single CUDA kernel renders the same image in $\approx 4.48\,\text{ms}$, $>100\times$ faster than MPI‑8 and $>700\times$ over the CPU baseline.

---

## 1. Problem and model
**Set** 

Mandelbrot points satisfy $z_{n+1}=z_n^2+c$ with $z_0=0$, $c=a+ib$. In reals:

$$
\begin{aligned}
 x_{n+1}&=x_n^2-y_n^2+a,\\
 y_{n+1}&=2x_n y_n+b,\qquad (x_0,y_0)=(0,0).
\end{aligned}
$$

**Escape test** 

Stop when $x_n^2+y_n^2>4$ because we know that necesarly leads to divergence. If no escape within `depth`, mark as inside.

**Pixel mapping**

 For domain $[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$ and image $(w,h)$:

$$
 x_{\text{inc}}=\frac{x_{\max}-x_{\min}}{w-1},\quad y_{\text{inc}}=\frac{y_{\max}-y_{\min}}{h-1}.
$$

**Coloring** 

If escape at iteration $i<\text{depth}$ then `color = i mod 255`, else `255`.

---

## 2. CPU baseline
Simple double loop over pixels. Each calls `xy2color(a,b,depth)` that iterates until escape or `depth`.

```c
unsigned char xy2color(double a, double b, int depth){
  double x=0.0, y=0.0;
  for(int i=0;i<depth;++i){
    double x2=x*x, y2=y*y;
    if(x2+y2>4.0) return (unsigned char)(i%255);
    double t=x; x=x2-y2+a; y=2.0*t*y+b;
  }
  return 255;
}
```

**Build**
```bash
gcc mandel.c -o mandel -O3 -march=native -mtune=native -ffast-math -lm
```

**Observed time.** $2.312\,\text{s}$ for $1000\times1000$, `depth=10000`.

---

## 3. MPI v1 — static stripes
**Partition** 

1D block decomposition over rows: each rank computes $\lfloor h/p\rfloor$ consecutive rows. Rank 0 gathers and writes.

<p align="center">
  <img src="/assets/img/HPC_mandelbrot/MPI1.png" alt="1D blocks" width="70%">
</p>

**Issue** 

Load is not uniform. Central rows intersect the set and perform more iterations. Peripheral rows escape early. With static stripes, some ranks finish early and idle.

**Results**

| Cores | Time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 2.312 | 1.00 | 1.000 |
| 2 | 1.190 | 1.94 | 0.972 |
| 4 | 1.189 | 1.95 | 0.486 |
| 8 | 0.989 | 2.34 | 0.292 |

**Build**
```bash
mpicc mandel_parallel_1.c -o mandel_mpi_static -O3 -march=native -mtune=native -ffast-math -lm
```

---

## 4. MPI v2 — Master–Worker (dynamic load balancing)
**Design** 

Rank 0 is the master. Workers repeatedly:

1. receive a task id for `nb_lines` rows;
2. compute a chunk;
3. send results back; and
4. continue requesting new tasks until receiving a termination signal from the master.

<p align="center">
  <img src="/assets/img/HPC_mandelbrot/MPI2.png" alt="Master Worker">
</p>

**Why it works** 

Tasks are small and reassigned as soon as a worker finishes, so fast ranks take more chunks. No prior knowledge of heavy rows is required. On shared clusters (e.g., Grid5000) it also absorbs heterogeneity.

**Efficiency bound** 

With $p$ total ranks and one non‑computing master:

$$
E\;=\;\frac{S}{p}\;\approx\;\frac{p-1}{p}.
$$

**Results.**

| Cores | Workers | Time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 2 | 1 | 2.377 | 0.973 | 0.486 |
| 4 | 3 | 0.793 | 2.92 | 0.729 |
| 9 | 8 | 0.345 | 6.70 | 0.837 |

![Comparison](/assets/img/HPC_mandelbrot/Speedup_Efficiency.png)

**Build.**
```bash
mpicc mandel_parallel_2.c -o mandel_mpi_dyn -O3 -march=native -mtune=native -ffast-math -lm
```

**Notes.** Choose `nb_lines` to balance message rate and latency. Coarser chunks reduce communication; finer chunks improve balance.

---

## 5. CUDA — one thread per pixel
**Kernel mapping**

 2D grid, 2D blocks. Each thread computes one pixel and writes `out[i*w + j]`.

```cuda
__device__ __forceinline__ unsigned char xy2color(double a,double b,int depth){
  double x=0.0, y=0.0;
  for(int i=0;i<depth;++i){
    double x2=x*x, y2=y*y;
    if(x2+y2>4.0) return (unsigned char)(i & 255);
    double t=x; x=x2-y2+a; y=2.0*t*y+b;
  }
  return 255;
}

__global__ void mandelKernel(unsigned char* out,int w,int h,
                             double xmin,double ymin,double incx,double incy,
                             int depth){
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  if(i>=h || j>=w) return;
  double a = xmin + j*incx;
  double b = ymin + i*incy;
  out[i*w + j] = xy2color(a,b,depth);
}
```

**Launch** 

Square blocks; sweep to tune occupancy. We tested $4,8,16,32$ and observed a minimum at $8\times8$ on the target GPU.

<p align="center">
  <img src="/assets/img/HPC_mandelbrot/BlockSize.png" alt="1D blocks" width="60%">
</p>

**Build**
```bash
nvcc mandel_gpu.cu -o mandel_gpu --generate-code arch=compute_60,code=sm_60 -O3
```

**Times and throughput**

| Config | Depth | Kernel time | Throughput |
|:--|--:|--:|--:|
| $800\times800$ | 10000 | $0.626\,\text{ms}$ | $\approx 1022$ Mpix/s |
| $1024\times1024$ | 10000 | $0.447\,\text{ms}$ | $\approx 2346$ Mpix/s |
| $2048\times2048$ | 10000 | $2.01\,\text{ms}$ | $\approx 2087$ Mpix/s |
| $4096\times4096$ | 10000 | $9.30\,\text{ms}$ | $\approx 1804$ Mpix/s |

**Depth sweep @ $800\times800$** 

$\{20000,50000,100000\}$ iterations yield $\{0.633,0.601,0.641\}\,\text{ms}$. Runtime is flat, indicating heavy early‑escape. Validate that `depth` is applied as a hard cap and not always reached.

---

## 6. Average iterations per pixel
Goal: compute

$$
\text{avg iters} = \frac{1}{w\,h} \sum_{\text{pixel}} \text{iters}.
$$

### A. Global atomic accumulation
Each thread atomically adds its local iteration count to a single global counter.

```cuda
__global__ void mandel_atomic(unsigned char* img, int w,int h,
                              double xmin,double ymin,double incx,double incy,
                              int depth, unsigned long long* sum){
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  if(i>=h || j>=w) return;
  // compute iters and color ...
  atomicAdd(sum, (unsigned long long)iters);
}
```
**Tradeoff**

 Correct but contended. Global atomics serialize updates making it a bit slow.

### B. Block‑local shared accumulation
Keep a shared `counter_block` per block. Threads `atomicAdd` to shared memory, then one global `atomicAdd` per block.

```cuda
extern __shared__ unsigned int counter_block[]; // single int via size==sizeof(int)
__global__ void mandel_shared(unsigned char* img, /*...*/, unsigned int* sum){
  if(threadIdx.x==0 && threadIdx.y==0) counter_block[0]=0;
  __syncthreads();
  // ... compute iters
  atomicAdd(&counter_block[0], (unsigned int)iters);
  __syncthreads();
  if(threadIdx.x==0 && threadIdx.y==0) atomicAdd(sum, counter_block[0]);
}
```

This implementation is faster.

### C. Full parallel reduction

First we write the iteration count for each pixel on an array `d_counter`.

Then a 1D kernel reduces a strided segment per block in shared memory. Host loop multiplies `stride` by `TPB` each pass until `blocks == 1`. `TPB = BLOCK_SIZE_X * BLOCK_SIZE_Y`.

```cuda
__global__ void reduceKernel(unsigned int* d_counter, size_t N, size_t stride){
  extern __shared__ unsigned int s_data[];  // size = TPB * sizeof(unsigned int)
  unsigned int tid = threadIdx.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + tid;
  size_t idx = stride * i;

  s_data[tid] = (idx < N) ? d_counter[idx] : 0u;
  __syncthreads();

  for(unsigned int s=1; s<blockDim.x; s<<=1){
    if((tid % (s<<1)) == 0) s_data[tid] += s_data[tid + s];
    __syncthreads();
  }

  if(tid==0){
    d_counter[stride * (size_t)blockIdx.x * blockDim.x] = s_data[0];
  }
}
```

Host-side loop:
```c
const unsigned int TPB = BLOCK_SIZE_X * BLOCK_SIZE_Y;
size_t N = (size_t)w * (size_t)h;
size_t stride = 1;
size_t blocks = (N + TPB - 1) / TPB;
while (1) {
  size_t shmem = TPB * sizeof(unsigned int);
  reduceKernel<<<blocks, TPB, shmem>>>(d_counter, N, stride);
  if (blocks == 1) break;
  stride *= TPB;
  blocks = (N + stride*TPB - 1) / (stride*TPB);
}
unsigned int total_iters = 0;
cudaMemcpy(&total_iters, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
float avg_iter = (float)total_iters / (float)(w*h);
```

**Notes** 

The stride progression collapses the array geometrically. One shared-memory reduction per block produces one global write per block per pass.

This implementation is the fastest.

---

## 7. GPU vs MPI
For $1000\times1000$, `depth=10000`:

| Platform | Workers | Time (s) | Speedup vs CPU |
|:--|--:|--:|--:|
| MPI dynamic | 8 | $0.345$ | $\approx 6.70$ |
| GPU kernel  | 3584 cores | $0.00448$ | $\approx 516$ |

The GPU kernel is $\sim110\times$ faster than MPI‑8 and $>700\times$ vs the sequential baseline on this workload.

**Why GPU dominates**
1. Massive parallelism: thousands of resident threads keep Streaming Multiprocessors busy.
2. Minimal synchronization: no inter‑thread communication between pixels.
3. Hardware scheduling: warps are swapped to hide latency.
4. Bandwidth: GPU DRAM bandwidth $\gg$ CPU socket bandwidth; writes are coalesced.

**Compute divergence within warps** 

Present near mandelbrot's set boundaries but limited because adjacent pixels often share iteration counts. Impact is modest.

---

## 8. Energy sketch
Rough model: GPU $\approx 250\,\text{W}$, CPU core $\approx 65\,\text{W}$.
- GPU: $4.48\,\text{ms} \times 250\,\text{W} \approx 1.12\,\text{J} \approx 0.31\,\text{mWh}$.
- MPI‑8: $0.345\,\text{s} \times 8\times65\,\text{W} \approx 179\,\text{J} \approx 49.7\,\text{mWh}$.
GPU looks $\mathcal{O}(10^2)$ more energy efficient on this task.

---

## References
- R. L. Devaney, *The Fractal Geometry of the Mandelbrot Set*.
- Mark Harris, *Optimizing Parallel Reduction in CUDA*. NVIDIA Developer Technology.
