# Solution guide: factorized tensor-ring inference kernel

**Audience:** an AI coding agent (e.g. Claude Code) with shell and Jupyter access on the supplied NVIDIA GPU machine. The agent implements the solution, runs it, and drafts the report for a human submitter who must be able to defend every line in a 45-minute technical discussion.

**Status of the claims in this guide:**
- The FLOP, byte and shape numbers were derived by hand from the index algebra in `IMPLEMENTATION.md` and `reference.py`.
- The kernel design and code are **unexecuted sketches**; no GPU was available when this guide was written.
- Anything labeled *hypothesis* must be confirmed or refuted by measurement.
- The notebook's verification cells (§6 and §7) are what make the solution trustworthy, not this document.

Source documents, in order of authority: [README.md](README.md), [IMPLEMENTATION.md](IMPLEMENTATION.md), [BENCHMARK_NOTES.md](BENCHMARK_NOTES.md), [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md), then the code ([reference.py](src/factorized_inference/reference.py), [submission.py](src/factorized_inference/submission.py), [benchmark.py](benchmarks/benchmark.py), [tests/](tests/)). [README-fa-detailed.md](README-fa-detailed.md) is good GPU background, but see the errata in §4.4.

---

## 0. Plan at a glance

1. **Environment:** create the venv on the GPU box, confirm `import triton` works, install Jupyter, and register the venv as a kernel.
2. **Profile the reference first:** do this in the notebook and from the harness trace. Write down the token-1 bottleneck with numbers before designing anything.
3. **Implement** the recommended kernel (§5), a fused **B→A→C** tensor-ring kernel in Triton:
   - One program per (token, output mode `q`, k-group); every intermediate stays on-chip.
   - A deterministic split-K reduction handles small token counts.
   - The source lives in `%%writefile` cells of `solution.ipynb`, which generate the package files the harness imports.
4. **Verify in three layers:** a pure-PyTorch float64 emulation of the kernel's loop structure, then Triton against the dense oracle, then `pytest` with no skips.
5. **Tune lightly:** SPLIT_K rule, `num_warps`, `num_stages`, and optionally the v2 variant. Freeze the configuration and disclose that you tuned it.
6. **Run the exact required commands.** Generate every table and trace summary from the raw JSON.
7. **Add supplementary evidence:**
   - Kernel presence in all five cases.
   - A rank-16 trace.
   - Repeat runs.
   - Ablations.
   - Optional: cold-L2 and CUDA-graph sections, clearly separated from the required results.
8. **Write `REPORT.md`** from measured numbers only, disclose AI use, log GPU hours, and **stop the GPU instance**.

---

## 1. Non-negotiable rules

The agent must satisfy all of these. Each one comes from the assignment documents.

| # | Rule | Source |
|---|---|---|
| R1 | Never construct or retain the complete dense weight, even temporarily, in any layout or precision, in the optimized path. Only baseline and oracle code are exempt. Permuted copies of individual cores are fine but must be reported. | README, IMPLEMENTATION |
| R2 | The custom kernel (Triton, CUDA C++ or CuTe DSL) performs a substantive part of the factorized math in **all five** CUDA cases: rank 8 × tokens {1, 8, 32} and rank 16 × tokens {1, 32}. `torch.compile` or a CUDA-graph wrapper alone does not count. | README, IMPLEMENTATION |
| R3 | Do not mutate `x` or the cores. One prepared callable must handle changing values and token counts. Two prepared callables with different weights must stay independent, so use no global weight caches. | IMPLEMENTATION, tests |
| R4 | Output is `[tokens, out_features]` on the input's device with the input's dtype: FP16 on GPU, FP32 for CPU checks. It must be usable on the caller's current stream. Document any output-buffer reuse (recommended: none). | IMPLEMENTATION |
| R5 | Required measurements use **no CUDA-graph capture**. If you use `torch.compile`, disable its cudagraph modes and record the options. | BENCHMARK_NOTES |
| R6 | Do not change the harness, tests, workloads or tolerances (FP16 `atol=rtol=0.02`, FP32 `1e-4`). Disclose any change you make anyway. | IMPLEMENTATION |
| R7 | The headline metric is **CUDA-event stream median**. Also report host median, preparation, first call, and the memory breakdown for every case and method. Report max abs, mean abs and relative-L2 error. | README, REPORT_TEMPLATE |
| R8 | Explain the token-1 profiler evidence and show where the custom kernel appears in the trace. | README |
| R9 | Cite the source and license of any adapted code. Share only code you are authorized to share. Disclose AI use and what was independently verified. | README, IMPLEMENTATION |
| R10 | A CPU fallback to `tr_forward_reference` is allowed. Beating dense is **not** required; a correct kernel with a well-explained negative result passes. | README |

---

## 2. Deliverables and repository layout

```text
solution.ipynb                              # THE solution: design, kernel source, verification, measurements, analysis
src/factorized_inference/tr_kernels.py      # generated by solution.ipynb (%%writefile): Triton kernels + launch wrapper
src/factorized_inference/submission.py      # generated by solution.ipynb (%%writefile): prepare_optimized / tr_forward_optimized
REPORT.md                                   # completed REPORT_TEMPLATE.md (same headings), numbers generated from JSON
results/rank8.json, results/rank16.json     # REQUIRED raw results (exact README commands)
results/environment.txt                     # REQUIRED pip freeze
results/supp_*.json                         # supplementary runs (labelled; never mixed with required tables)
traces/rank8/*.json                         # REQUIRED token-1 profiler traces
traces/rank16/*.json, traces/*.nsys-rep, traces/*.ncu-rep   # optional supplementary
```

- Keep `benchmarks/`, `tests/` and `reference.py` unchanged.
- **Gotcha:** `.gitignore` excludes `results/`, `traces/`, `*.nsys-rep` and `*.ncu-rep`, but these are required submission artifacts. Use `git add -f` for them, or include them explicitly in the ZIP. Record the commit hash or the ZIP's SHA-256 in the report.

---

## 3. The notebook: why and how it works with the harness

The solution must be a Jupyter notebook, and the harness has constraints the notebook has to respect:

- **The harness runs every (method, token) case in a fresh subprocess.** That subprocess imports `factorized_inference.prepare_optimized` from `src/`. Code that exists only in notebook memory is invisible to it. The same applies to `pytest`.
- **Therefore the notebook is the single source of truth, and it writes the modules with `%%writefile`.** Do not hand-edit the generated `.py` files; edit the cell and re-run it. Put the comment `# GENERATED by solution.ipynb — edit the notebook, not this file` at the top of each generated file.
- **Location:** put `solution.ipynb` at the repository root and start Jupyter from the root, so relative paths in `%%writefile src/...` and in the harness commands resolve. Assert `Path("benchmarks/benchmark.py").exists()` in the first cell.
- **Always launch subprocesses with the notebook's interpreter.** Use `subprocess.run([sys.executable, "benchmarks/benchmark.py", ...], check=True)` or `!{sys.executable} -m pytest -q`. A bare `!python` may resolve to a different environment.
- **Reload semantics:** after re-running a `%%writefile` cell, call `importlib.reload(tr_kernels)` and then `importlib.reload(submission)`. The package attribute `factorized_inference.prepare_optimized` stays **stale** because `__init__.py` bound the old function. In the notebook, always call `submission.prepare_optimized`, or restart the kernel. Subprocesses such as the harness and pytest always see the fresh files.
- **Measurement hygiene:**
  - Run the required benchmark cells while no other notebook cell is using the GPU.
  - Check `nvidia-smi` for other processes and print them.
  - Unset any ablation environment variables (§8.3) before the required runs, and print the environment to prove it.
  - The notebook's own CUDA context does not affect the child processes' PyTorch memory counters, but concurrent GPU work does distort timing.
- **Dependencies:** `jupyterlab` or `notebook` and `ipykernel` are not pinned by `pyproject.toml`.
  - Install them into the venv and register the kernel with `python -m ipykernel install --user --name factorized-venv`.
  - Mention in the report that they are notebook-only tools; they will show up in `pip freeze`.
  - Prefer the standard library over pandas or matplotlib. If you add plotting, install it and disclose that too.
- **Triton availability:** Linux CUDA wheels of `torch==2.14.0` normally pull in a matching `triton`.
  - Verify with `import triton; triton.__version__`.
  - If it is missing, install the version that `pip show torch` lists under Requires, and disclose it.
- **Remote GPU access:** use Jupyter over an SSH tunnel (`ssh -L 8888:localhost:8888 ...`) or VS Code Remote-SSH with the notebook open remotely.
- **Final state:**
  - Do "Restart kernel & Run All", or run `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 solution.ipynb`.
  - Commit the notebook **with outputs**, because reviewers read them.
  - Optionally also export `solution.html`.

---

## 4. The problem in numbers

Notation: inputs have modes `(I, J, K) = (8, 12, 20)` with indices `i, j, k`, giving `in = 1920`. Outputs have modes `(P, Q, S) = (12, 10, 24)` with indices `p, q, r`, giving `out = 2880`. The size of the third output mode is written `S` to avoid a clash with the rank. The rank is `ρ` (8 or 16), and the ring indices `a, b, c` run over `[0, ρ)`.

```text
A[a,p,i,b]  B[b,q,j,c]  C[c,r,k,a]          (core shape [ρ, out_mode, in_mode, ρ], all contiguous)
y[t,p,q,r] = Σ_{i,j,k,a,b,c} x[t,i,j,k] · A[a,p,i,b] · B[b,q,j,c] · C[c,r,k,a]
flattening: in index = (i·J + j)·K + k,  out index = (p·Q + q)·S + r     (row-major, as in reference.py)
```

### 4.1 Storage

| | ρ = 8 | ρ = 16 |
|---|---|---|
| Factor params `ρ²(PI + QJ + SK) = 696ρ²` | 44,544 (87.0 KiB FP16) | 178,176 (348.0 KiB) |
| Dense params `2880·1920` | 5,529,600 (10.55 MiB FP16) | same |
| Compression | 124× | 31× |
| Recommended packed state (`Bp` + `Cp` + `A` snapshot = 696ρ²) | 87.0 KiB | 348.0 KiB |

### 4.2 FLOPs per token (2 × multiply-adds)

| Path | Stage 1 | Stage 2 | Stage 3 | Total ρ=8 | Total ρ=16 | vs dense (8 / 16) |
|---|---|---|---|---|---|---|
| Dense GEMV | | | | 11.06 M | 11.06 M | 1× |
| Reference order A→B→C | `x·A` over i: 46,080ρ² | `·B` over (j,b): 57,600ρ³ | `·C` over (k,a,c): 115,200ρ² | 39.81 M | 277.22 M | 3.6× / 25.1× |
| **B→A→C (recommended)** | `x·B` over j: 38,400ρ² | `·A` over (i,b): 38,400ρ³ | `·C` over (k,a,c): 115,200ρ² | **29.49 M** | **196.61 M** | 2.67× / 17.8× |

- Among the six "contract `x` with one core, then another" orders, B→A→C has the fewest FLOPs. The reference order sits between the extremes; A→C is the worst.
- Pre-contracting two cores is allowed as partial precomputation, but it expands storage a lot:
  - `A·B` is 11,520ρ² elements, which is 53% of dense at ρ=16.
  - `B·C` and `A·C` are larger than dense at ρ=16.
  - The only FLOP gain is `A·B` at ρ=16: 165 vs 197 MFLOP/token. At ρ=8 it adds FLOPs (41 vs 29.5).
  - Mention it as a trade-off you considered, not as the design.
- Cross-check these numbers in the notebook with `torch.utils.flop_counter.FlopCounterMode` over the reference and over an einsum implementation of B→A→C.

### 4.3 Intermediates per token (elements)

| | ρ = 8 | ρ = 16 | ρ = 16, T = 32 (FP16) |
|---|---|---|---|
| Reference I1 `[t,j,k,a,p,b]` = 2880ρ² | 184,320 | 737,280 | 45.0 MiB |
| Reference I2 `[t,k,a,p,q,c]` = 2400ρ² | 153,600 | 614,400 | 37.5 MiB |
| B→A→C U `[t,i,k,b,q,c]` = 1600ρ² | 102,400 | 409,600 | 25.0 MiB |
| **Fused kernel (§5)**: global intermediates | **0** | **0** | split-K workspace only: `T·SPLIT_K·2880·4 B` |

`torch.einsum` also makes hidden `permute().contiguous()` copies of I1 and I2 between stages, so the reference's real traffic is higher than the table shows.

### 4.4 What these numbers predict (hypotheses to test)

- **The factorized operator does more arithmetic than dense.** Per token it needs 2.7–25× the FLOPs of dense. Its advantage is only the bytes of weight it stores and reads.
- **At T ≤ 32, dense is a bandwidth-bound GEMV** over a 10.55 MiB weight. On GPUs whose L2 is larger than that (A100 40 MB, H100 50 MB, L4 48 MB, L40S 96 MB), the harness's repeated calls keep the weight L2-resident. That is the warm-cache caveat in BENCHMARK_NOTES. *Hypothesis:* dense wins every required case on such GPUs. On small-L2 GPUs (A10G 6 MB, T4 4 MB), dense streams from DRAM and the gap narrows. Record the L2 size (`torch.cuda.get_device_properties(0).L2_cache_size`, if present) and interpret the results with it.
- **At T = 1, everything is roughly 5–30 µs**, so host dispatch and launch count matter as much as device time. *Hypothesis:* the reference's three `einsum` calls expand into several GEMM and copy kernels with idle gaps between them, and its stream time is close to its host dispatch time. Verify this in the trace (§8.2).
- **At ρ = 16, T = 32**, the reference moves tens of MiB of intermediates, so fusion should pay off most there. At the same time, the ρ³ term makes FLOPs matter.
- **A cold-cache break-even model** belongs in the report's system section. Compute it with the real GPU's specs, using the peak you actually achieved:

  ```text
  dense time   ≈ W_bytes / BW_dram            (11.06 MB / BW)
  factorized   ≈ T · F_fact / P_achieved      (+ launch overhead)
  T*           ≈ (W_bytes / BW_dram) · (P_achieved / F_fact)
  ```

**Errata in the Persian guides. Do not propagate these:**
- `README-fa-detailed.md` §16 says dense is "about 10.6B FLOP per token". The correct figure is `2·2880·1920 ≈ 11.06 MFLOP` per token.
- `README-fa.md` §3 suggests a kernel in which each output element sums directly over all of `i, j, k, a, b, c`. That costs `out·in·ρ³ ≈ 2.8 G` (ρ=8) to 22.6 G (ρ=16) products per token, which is effectively rebuilding W for every token. `README-fa-detailed.md` §18 already warns against this. Avoid it.

---

## 5. Recommended kernel: fused B→A→C ring contraction (Triton)

### 5.1 The idea in one paragraph

Contract in the order `x·B` (over `j`), then `·A` (over `i, b`), then `·C` (over `k, a, c`).

- Each Triton program owns **one token `t`, one output-mode index `q`, and a group of `k` values**.
- For each `k` it builds a small `U` tile on-chip from `x` and `B`.
- It then loops over the ring index `a`. For each `a` it does two small tensor-core dots, `A_a @ U` followed by `(…) @ C_{k,a}`, and accumulates `y[t, :, q, :]` in FP32 registers.
- No intermediate touches global memory.
- When `T` is small, `k` is split across programs to create parallelism. A tiny second kernel then sums the fixed-order FP32 partials, so the result is deterministic.
- The forward is **one launch** when SPLIT_K = 1 and two launches otherwise. The reference forward runs several GEMM and copy kernels.

### 5.2 The math, stage by stage (verify each identity in the notebook)

```text
U[t,k][(i,b), c]  = Σ_j   x[t,i,j,k] · B[b,q,j,c]                  stage 1  (CUDA-core FMAs; J = 12 is tiny)
V[t,k,a][p, c]    = Σ_(i,b) A[a,p,i,b] · U[t,k][(i,b), c]           stage 2  (tl.dot  [P × Iρ] @ [Iρ × ρ])
y[t][p, r]  +=      Σ_c   V[t,k,a][p, c] · C[c,r,k,a]               stage 3  (tl.dot  [P × ρ] @ [ρ × S]),  summed over a and k
```

Substituting back gives `Σ_{i,j,k,a,b,c} x·A·B·C`, which is exactly the definition. Work is not recomputed, because the programs partition `(t, q, k)` exactly:

- Stage 1 totals 19,200ρ²·T multiply-adds.
- Stage 2 totals 19,200ρ³·T.
- Stage 3 totals 57,600ρ²·T.

### 5.3 Layouts and persistent state (built once in `prepare_optimized`)

| Tensor | Layout | Built as | Size | Why |
|---|---|---|---|---|
| `x` | `[T, I, J, K]` (caller's, read-only) | view | — | Gathered as `x[t, i, j, k]`; `k` is fixed per iteration |
| `A_s` | `[ρ, P, I, ρ]` = `A[a,p,i,b]` | `A.contiguous().clone()` (snapshot) | 96ρ² | Rows `p`, columns `(i,b)` contiguous: stage-2 left operand for a fixed `a` |
| `Bp` | `[J, ρ, Q, ρ]` = `B[b,q,j,c]` | `B.permute(2,0,1,3).contiguous()` | 120ρ² | For fixed `(j, q)`: rows `b`, columns `c` contiguous |
| `Cp` | `[K, ρ, ρ, S]` = `C[c,r,k,a]` | `C.permute(2,3,0,1).contiguous()` | 480ρ² | For fixed `(k, a)`: rows `c`, columns `r` contiguous: stage-3 right operand |
| `out` | `[T, P·Q·S]` FP16 | `torch.empty` per call | 5,760·T B | Fresh each call; no reuse semantics to document |
| `ws` (only if SPLIT_K > 1) | `[T, SPLIT_K, P·Q·S]` FP32 | `torch.empty` per call | 11,520·T·SPLIT_K B | Deterministic split-K partials |

- These are permutations or copies of individual cores, with no expansion (R1). Report them as roughly 1× the factor bytes of extra resident state.
- Snapshot semantics: the prepared state reflects the cores at prepare time. Document this.
- If you would rather reference `A` in place to save 96ρ² elements, say so. The trade-off is a mixed state if a caller later mutated its cores in place.

### 5.4 Padding rules (Triton constraints)

`tl.arange` sizes must be powers of 2, and every `tl.dot` dimension must be at least 16 (check the installed Triton version's rule). Pad with **masked loads that use `other=0.0`**: padded lanes must hold exact zeros, or they contaminate the dot products.

```text
IP = next_pow2(I)                         # 8
BP = next_pow2(max(ρ, ceil(16 / IP)))     # ρ for defaults; ensures IP·BP ≥ 16 (rank-3 test → 8)
CP = max(16, next_pow2(ρ))                # 16 for both default ranks (ρ=8 wastes half)
PP = max(16, next_pow2(P))                # 16
SP = max(16, next_pow2(S))                # 32
```

- These rules make the kernel run on every CUDA shape in the tests, including `TRSpec((2,3,2),(2,4,3),3)` in `test_prepared_weights_are_independent`. That is better evidence than falling back to the reference for small shapes.
- Report padding efficiency, meaning useful MACs divided by issued MACs:
  - Stage 2: `P·I·ρ·ρ / (PP·IP·BP·CP)` = 37.5% at ρ=8 and 75% at ρ=16.
  - Stage 3: `P·ρ·S / (PP·CP·SP)` = 28% at ρ=8 and 56% at ρ=16.
- This waste is acceptable at T ≤ 32, where latency and not FLOPs dominates, but you must discuss it.

### 5.5 Kernel sketch (unexecuted; validate against the §7.1 emulation)

```python
import triton
import triton.language as tl


@triton.jit
def tr_ring_fused_kernel(
    x_ptr, a_ptr, bp_ptr, cp_ptr, dst_ptr,
    I: tl.constexpr, J: tl.constexpr, K: tl.constexpr,
    P: tl.constexpr, Q: tl.constexpr, S: tl.constexpr, RANK: tl.constexpr,
    IP: tl.constexpr, BP: tl.constexpr, CP: tl.constexpr, PP: tl.constexpr, SP: tl.constexpr,
    SPLIT_K: tl.constexpr, K_PER: tl.constexpr,
):
    pid = tl.program_id(0)                  # flattened (token, q); T is never a kernel argument
    g = tl.program_id(1)                    # k-group
    t = pid // Q
    q = pid % Q

    ib = tl.arange(0, IP * BP)              # stage-2 reduction axis, flattened (i, b)
    i_of = ib // BP
    b_of = ib % BP
    ib_ok = (i_of < I) & (b_of < RANK)
    c = tl.arange(0, CP)
    c_ok = c < RANK
    p = tl.arange(0, PP)
    p_ok = p < P
    s = tl.arange(0, SP)
    s_ok = s < S

    x_rows = x_ptr + t * (I * J * K) + i_of * (J * K)                                   # [IP*BP]
    b_tile = bp_ptr + b_of[:, None] * (Q * RANK) + q * RANK + c[None, :]                 # [IP*BP, CP]
    b_mask = ib_ok[:, None] & c_ok[None, :]
    a_tile = a_ptr + p[:, None] * (I * RANK) + i_of[None, :] * RANK + b_of[None, :]      # [PP, IP*BP]
    a_mask = p_ok[:, None] & ib_ok[None, :]
    c_tile = cp_ptr + c[:, None] * S + s[None, :]                                        # [CP, SP]
    c_mask = c_ok[:, None] & s_ok[None, :]

    acc = tl.zeros((PP, SP), dtype=tl.float32)
    for kk in range(K_PER):
        k = g * K_PER + kk
        k_ok = k < K
        # Stage 1: U[(i,b), c] = sum_j x[t,i,j,k] * B[b,q,j,c]
        u = tl.zeros((IP * BP, CP), dtype=tl.float32)
        for j in range(J):
            xv = tl.load(x_rows + j * K + k, mask=(i_of < I) & k_ok, other=0.0).to(tl.float32)
            bv = tl.load(b_tile + j * (RANK * Q * RANK), mask=b_mask, other=0.0).to(tl.float32)
            u += xv[:, None] * bv
        u = u.to(tl.float16)
        for a in range(RANK):
            # Stage 2: V[p, c] = sum_(i,b) A[a,p,i,b] * U[(i,b), c]
            av = tl.load(a_tile + a * (P * I * RANK), mask=a_mask, other=0.0)
            v = tl.dot(av, u)                                                            # [PP, CP] fp32
            # Stage 3: y[p, r] += sum_c V[p, c] * C[c,r,k,a]
            cv = tl.load(c_tile + (k * RANK + a) * (RANK * S), mask=c_mask & k_ok, other=0.0)
            acc += tl.dot(v.to(tl.float16), cv)                                          # [PP, SP] fp32

    offs = p[:, None] * (Q * S) + q * S + s[None, :]
    mask = p_ok[:, None] & s_ok[None, :]
    if SPLIT_K == 1:
        tl.store(dst_ptr + t * (P * Q * S) + offs, acc.to(tl.float16), mask=mask)
    else:
        tl.store(dst_ptr + (t * SPLIT_K + g) * (P * Q * S) + offs, acc, mask=mask)


@triton.jit
def tr_splitk_reduce_kernel(ws_ptr, out_ptr, N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for g in range(SPLIT_K):                               # fixed order: deterministic
        acc += tl.load(ws_ptr + (t * SPLIT_K + g) * N + offs, mask=m, other=0.0)
    tl.store(out_ptr + t * N + offs, acc.to(tl.float16), mask=m)
```

The offset formulas below are exactly what §7.1 checks:

```text
x[t,i,j,k]      = t·IJK + i·JK + j·K + k
A[a,p,i,b]      = a·PIρ + p·Iρ + i·ρ + b
Bp[j,b,q,c]     = j·ρQρ + b·Qρ + q·ρ + c
Cp[k,a,c,r]     = (k·ρ + a)·ρS + c·S + r
y[t,p,q,r]      = t·PQS + p·QS + q·S + r
```

Rows with `k ≥ K` in the last k-group are masked to zero, so SPLIT_K does not have to divide K. The dispatch rule below still restricts it to divisors so that fewer variants compile.

### 5.6 Host wrapper (sketch)

```python
def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


class _FusedTR:
    TARGET_PROGRAMS_PER_SM = 2          # tuned in the notebook; disclose

    def __init__(self, cores, spec):
        a, b, c = cores
        (I, J, K), (P, Q, S), R = spec.input_modes, spec.output_modes, spec.rank
        self.cores, self.spec = cores, spec              # kept for the non-fp16 fallback
        self.a = a.contiguous().clone()
        self.bp = b.permute(2, 0, 1, 3).contiguous()
        self.cp = c.permute(2, 3, 0, 1).contiguous()
        ip = _pow2(I)
        self.meta = dict(I=I, J=J, K=K, P=P, Q=Q, S=S, RANK=R, IP=ip,
                         BP=_pow2(max(R, -(-16 // ip))), CP=max(16, _pow2(R)),
                         PP=max(16, _pow2(P)), SP=max(16, _pow2(S)))
        self.K, self.Q, self.n_out = K, Q, P * Q * S
        self.sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        self.splits = [d for d in range(1, K + 1) if K % d == 0]

    def split_for(self, T: int) -> int:
        want = math.ceil(self.TARGET_PROGRAMS_PER_SM * self.sms / (T * self.Q))
        return next((d for d in self.splits if d >= want), self.K)

    def __call__(self, x):
        # validate: 2D, x.shape[1] == in_features, same device; non-fp16 → reference fallback
        x = x.contiguous()                   # no-op for the required contiguous inputs
        T = x.shape[0]
        out = torch.empty((T, self.n_out), device=x.device, dtype=torch.float16)
        if T == 0:
            return out
        split = self.split_for(T)
        if split == 1:
            tr_ring_fused_kernel[(T * self.Q, 1)](x, self.a, self.bp, self.cp, out,
                                                  **self.meta, SPLIT_K=1, K_PER=self.K)
        else:
            ws = torch.empty((T, split, self.n_out), device=x.device, dtype=torch.float32)
            tr_ring_fused_kernel[(T * self.Q, split)](x, self.a, self.bp, self.cp, ws,
                                                      **self.meta, SPLIT_K=split, K_PER=self.K // split)
            tr_splitk_reduce_kernel[(triton.cdiv(self.n_out, 1024), T)](ws, out, N=self.n_out,
                                                                         SPLIT_K=split, BLOCK=1024)
        return out
```

`prepare_optimized(cores, spec)` should return `_FusedTR(cores, spec)` when the cores are CUDA FP16, and otherwise `lambda x: tr_forward_reference(x, cores, spec)`. Keep `tr_forward_optimized` exported, because `__init__.py` imports it. It can call `prepare_optimized(cores, spec)(x)`.

**Hot-path rules:**
- Do no `torch.cuda.synchronize()`, `.item()`, `print` or `torch.isfinite` asserts inside `__call__`.
- Never make `T` a kernel argument or a constexpr. It lives only in the grid, which avoids Triton's integer specialization, where ints equal to 1 or divisible by 16 trigger recompiles. If you later add a token tile (`TT`) and need `T` for masking, mark it `do_not_specialize`.
- Keep every static value in the precomputed `self.meta`.
- Do not use `@triton.autotune` at call time.
- Triton launches on `torch.cuda.current_stream()`, and the caching allocator is stream-ordered. That satisfies R4.

**JIT cost:**
- Lazy compilation lands in `first_call_ms`, which the harness reports separately, so this is allowed.
- Report the first call with a cold Triton cache (a fresh `TRITON_CACHE_DIR`) and with a warm one.
- Alternatively, compile eagerly in `prepare_optimized` by launching every SPLIT_K variant on a dummy input. Disclose that the cost then shows up in `preparation_ms`.

### 5.7 Precision

- Loads are FP16. Stage 1 accumulates in FP32 and is rounded to FP16 as a `tl.dot` operand. Stages 2 and 3 use FP16 tensor-core inputs with FP32 accumulation. The accumulation over `a` and `k` happens in FP32 registers or in the FP32 split-K workspace. One final cast produces FP16.
- This matches the reference's behaviour, which also rounds I1 and I2 to FP16, so expect errors of similar magnitude. The outputs are roughly N(0,1) because of the `make_cores` scaling.
- Do not pass FP32 operands to `tl.dot`. Its default is TF32, and the harness disables TF32 for PyTorch. For CUDA FP32 inputs, fall back to the reference, or use `input_precision="ieee"` and document it.

### 5.8 Optimizations to try only after it is correct and measured

Keep each optimization behind a flag, measure it (§8.3), and keep it only if it helps. Every ablation is useful material for the discussion.

1. **Tuning knobs:**
   - SPLIT_K rule: `TARGET_PROGRAMS_PER_SM` ∈ {1, 2, 4}, with SPLIT_K restricted to divisors of K.
   - `num_warps` ∈ {1, 2, 4}: 16-row dots usually prefer 1–2.
   - `num_stages` ∈ {1, 2, 3}: software-pipeline the `a` loop's loads.
2. **Wider N tile (TT tokens × QT values of q per program).** The column axis of `U` becomes `(tt, q, c)`, still loaded as a pure 2-D gather. After stage 2, `tl.reshape(v, [PP·TT·QT, CP])` is a legal row-major reshape. The accumulator gets rows `(p, tt, q)`. Each `A_a` tile is then reused TT·QT times, which matters at T = 32, where A is re-read from L1/L2 by every program.
3. **v2, vectorize the ring index `a`.** Pack `Ap = A.permute(1,0,2,3)` as `[P, ρ, I, ρ]`. Per `k`, compute one dot `[(p,a) × (i,b)] @ [(i,b) × c]`, reshape the result to `[p, (a,c)]`, and do one dot with `Cp[k]` viewed as `[(a,c) × r]`. The loop trip count drops from K_PER·ρ to K_PER, and the dots get bigger. The costs are a layout conversion when the dot result is reshaped, and a 64 KiB A tile at ρ=16, which raises shared-memory and register pressure. Combining v2 with TT or QT greater than 1 needs `tl.permute`.
4. **Host overhead.** If T = 1 is host-bound, which you can tell when stream time is about equal to host time and far above the trace's kernel time, quantify the floor by timing an empty Triton launch and a bare `F.linear`. Go further (a C++ launcher, or CUDA graphs in the optional section) only if time permits.

### 5.9 Plan B, if the fused kernel is not correct within a reasonable time

Use a **Triton kernel for stage 2 only** (the dominant contraction) in B→A→C order:

1. Stage 1 is `torch.matmul` on strided views: `x.view(T·I, J, K).transpose(1, 2) @ Bp.view(J, ρ·Q·ρ)`, which gives `U[t,i,k,(b,q,c)]`.
2. The Triton stage-2 kernel reads `U` with per-index offsets (no permute copies) and writes `V` directly in the `[t, p, q, k, a, c]` layout.
3. Stage 3 is then a plain `torch.matmul(V.view(T·P·Q, K·ρ²), Cp.view(K·ρ², S))`.

This still satisfies R2: the custom kernel performs the largest contraction in every case. Report honestly that the intermediates are materialized.

### 5.10 Approaches to avoid

- **A six-loop, one-thread-per-output direct kernel** (§4.4 errata).
- **Precomputing `A⊗B`, `B⊗C` or W tiles covering all of W.** This is storage expansion that approaches or exceeds dense; for ρ=16, `B·C` alone would be 2.7× dense.
- **CUDA C++ WMMA from scratch as the first attempt.** It is slower to iterate on. Use it only if Triton is unavailable.
- **Atomics on FP16 outputs.** They are nondeterministic and lose precision.
- **Any module-level cache keyed by shape or `id(cores)`.** It breaks independence, and ids get reused.

---

## 6. Notebook blueprint (`solution.ipynb`)

Each section starts with a short markdown cell that explains *why*. Code cells stay small; long code goes in `%%writefile` modules.

| § | Section | Contents | Outputs kept in the notebook |
|---|---|---|---|
| 0 | Title and summary | Two or three sentences of findings (written last); how to run it; links to REPORT.md and artifacts | — |
| 1 | Environment | Assert repo root; print `sys.executable`, Python, torch, CUDA, triton, driver (`nvidia-smi`), GPU name, SM count, L2 size, clocks; `nvidia-smi` process list; GPU-session start timestamp | Environment table |
| 2 | Problem analysis | Recompute every table in §4 with code; `FlopCounterMode` cross-check; break-even model using the GPU's datasheet bandwidth and FP16 peak (cite the source) | Tables |
| 3 | Reference profiling | In-process `torch.profiler` of reference and dense at T=1 (and all five cases): kernel list, counts, device time vs wall time; bottleneck statement | Kernel tables |
| 4 | Design | Markdown: §5.1–5.4 in your own words; index/offset table; ownership diagram (ASCII); padding efficiency; what is persistent and what is per call | — |
| 5 | Emulation | Pure-PyTorch float64 `emulate_fused` that mirrors the kernel's loops (§7.1); asserts against the dense oracle on several specs, ranks and SPLIT_K values | Pass/fail table |
| 6 | Kernel source | `%%writefile src/factorized_inference/tr_kernels.py` | — |
| 7 | Integration | `%%writefile src/factorized_inference/submission.py`, then reload (§3) | — |
| 8 | Correctness matrix | §7.2 checks in-process | Error table (max abs, mean abs, rel-L2, worst `|err|/(atol+rtol|exp|)`) |
| 9 | Test suite | `!{sys.executable} -m pytest -q -rA` must show **0 skipped** on the GPU | pytest summary |
| 10 | Kernel presence and memory | Profile `prepare_optimized(...)(x)` for all five cases and assert the custom kernel name appears; persistent-state inventory; allocation deltas prove nothing near 10.55 MiB is allocated | Tables |
| 11 | Tuning | Small in-process sweep (§8.3); chosen config; note that the constants in §6 were set from it | Sweep table |
| 12 | Required runs | Exact README commands via `subprocess.run([sys.executable, ...])`; then `pip freeze > results/environment.txt` | Harness stdout |
| 13 | Results | Tables generated from the JSON (§8.1): template table, preparation/first-call, memory breakdown, speedups, event-sample spread, correctness | Markdown tables |
| 14 | Profiler evidence | Parse `traces/rank8/*_tokens1.json` for all three methods (§8.2); show the custom kernel row | Tables |
| 15 | Supplementary | Rank-16 trace run, repeat runs, ablations; optional nsys/ncu, cold-L2 study, CUDA-graph study; each clearly labelled | Tables |
| 16 | Report | Write `REPORT.md`: tables injected from §13–§15, narrative written after reading them | File written |
| 17 | Close-out | GPU-session end timestamp and hours; artifact checklist (§12) printed with ✓/✗; reminder to stop the instance | Checklist |

Build and iterate in this order: 1–5, then 6–9 until green, then 10–11, freeze the config, then 12–17. Finish with a full Restart & Run All.

---

## 7. Correctness protocol

### 7.1 Emulate before writing Triton

This validates the index algebra and padding logic on CPU in float64. The following mirrors the kernel's loops:

```python
def emulate_fused(x, cores, spec, split_k=1):
    A, B, C = (t.double() for t in cores)
    (I, J, K), (P, Q, S), R = spec.input_modes, spec.output_modes, spec.rank
    T = x.shape[0]
    X = x.double().reshape(T, I, J, K)
    Bp = B.permute(2, 0, 1, 3)                     # [J, b, q, c]
    Cp = C.permute(2, 3, 0, 1)                     # [k, a, c, r]
    k_per = -(-K // split_k)
    ws = torch.zeros(T, split_k, P, Q, S, dtype=torch.float64)
    for t in range(T):
        for q in range(Q):
            for g in range(split_k):
                acc = torch.zeros(P, S, dtype=torch.float64)
                for k in range(g * k_per, min((g + 1) * k_per, K)):
                    U = torch.einsum('ij,jbc->ibc', X[t, :, :, k], Bp[:, :, q, :]).reshape(I * R, R)
                    for a in range(R):
                        V = A[a].reshape(P, I * R) @ U   # [p, c]
                        acc += V @ Cp[k, a]              # [p, r]
                ws[t, g, :, q, :] = acc
    return ws.sum(1).reshape(T, P * Q * S)
```

- Compare with `dense_forward(x.double(), materialize_dense_weight(double cores))` (oracle only) at a tolerance of about 1e-10.
- Specs to cover: the default modes at ρ ∈ {3, 8, 16}, `(2,3,2)/(2,4,3)` at ρ=3, and `(4,4,4)/(4,4,4)` at ρ ∈ {8, 16}.
- Use T ∈ {1, 2} and every divisor SPLIT_K of K.
- Also check the einsum B→A→C chain against the oracle.

### 7.2 Triton vs oracle (in-process, GPU)

- **Matrix:** default spec × ρ ∈ {8, 16} × T ∈ {1, 2, 3, 8, 17, 32, 33, 64} × seeds {0, 19, 23}, plus every test spec. Force every SPLIT_K value through a debug hook so that each compiled variant is checked.
- **Special inputs:**
  - `x = 0` must give exactly 0.
  - One-hot `x` at a few input indices must equal the matching column of the oracle's W.
  - All-ones input.
  - Scale 2.0, as in the tests.
  - Integer-valued small inputs, which make index permutations visible.
- **Interface checks:**
  - `x` and the cores are bit-identical after the call.
  - Output shape, dtype and device are correct, and the output is finite.
  - One prepared object gets varying T in sequence.
  - Two prepared objects are called interleaved.
  - Run under `with torch.cuda.stream(s):` without an extra sync, then `s.synchronize()` and check the result.
  - The same input twice with the same SPLIT_K gives bitwise-equal outputs (determinism).
- **Error metrics per case:** max abs, mean abs, rel-L2, and the headroom `max(|err| / (atol + rtol·|expected|))`, which must be below 1. Compare with the reference's errors under the same oracle.
- **Debugging aids:**
  - `CUDA_LAUNCH_BLOCKING=1`.
  - `TRITON_INTERPRET=1`, which runs Triton kernels on CPU through NumPy. It must be set **before** `import triton`, so use it in a subprocess.
  - Test tiny shapes first, and compare intermediates by temporarily storing `u` or `v` to a debug buffer.

### 7.3 Test suite

- `pytest -q -rA` on the GPU passes with **no skips**. This includes `test_required_cuda_workloads[...]` for all five cases and `test_prepared_weights_are_independent[cuda-...]`.
- Passing tests do not prove that R2 holds, because the placeholder also passes them. §6 step 10 (kernel presence) is what proves it.

---

## 8. Measurement and profiling protocol

### 8.1 Required runs (exact commands, same GPU, nothing else running)

```bash
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 8 --tokens 1,8,32 --output results/rank8.json --profile-dir traces/rank8
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 16 --tokens 1,32 --output results/rank16.json
python -m pip freeze > results/environment.txt
```

Run these from the notebook with `sys.executable`. Then generate the tables from the JSON; never type numbers by hand. The fields per method are:

- `host_synchronized_median_ms`
- `cuda_event_stream_median_ms` and `cuda_event_stream_samples_ms` (show min/max of the 5 samples)
- `preparation_ms`, `first_call_ms`
- `memory.{resident_after_warmup_allocated_bytes, extra_resident_allocated_bytes, preparation_and_warmup_peak_allocated_bytes, steady_peak_allocated_bytes, incremental_workspace_and_output_bytes, steady_peak_reserved_bytes}`
- `correctness.{max_absolute_error, mean_absolute_error, relative_l2_error}`

The template table uses these columns in MiB (2²⁰):

| Column | Field |
|---|---|
| Resident MiB | `resident_after_warmup_allocated_bytes` |
| Steady allocated peak MiB | `steady_peak_allocated_bytes` |
| Incremental workspace/output MiB | `incremental_workspace_and_output_bytes` |

Add a speedup table on CUDA-stream medians (optimized vs reference, optimized vs dense) and a second table with preparation, first call, extra resident, setup peak and reserved memory.

Expected persistent extra for the optimized method is roughly the Bp + Cp + A-snapshot bytes from §4.1 plus allocator rounding (blocks of 512 B). If the measured value differs, explain why. Expected incremental memory is the output plus, when SPLIT_K > 1, the workspace.

### 8.2 Profiler evidence (token 1)

The harness writes `traces/rank8/{dense,factorized_reference,factorized_optimized}_rank8_tokens1.json`. Each is a Chrome trace of three calls, each wrapped in `record_function(<method>)`.

1. Load `traceEvents`. First print the counter of the `cat` values, because category names vary across torch versions. Then filter the GPU kernels (typically `cat == "kernel"`) and the launches (`cuda_runtime` events such as `cudaLaunchKernel` or `cuLaunchKernel`).
2. Compute, per method and per call (divide the totals by 3):
   - The number of kernels and their names, grouped.
   - The sum of kernel durations (device busy).
   - The span from the first kernel's start to the last kernel's end.
   - The idle-gap fraction.
   - The CPU duration of the `user_annotation` span (host time).
   - The number of copy or elementwise kernels, meaning the hidden permutes.
3. **Bottleneck statement for the reference:** say how much of its time is device compute, how much is copies, and how much is idle gaps and host dispatch, citing the numbers. Compare the host span with `cuda_event_stream_median_ms`.
4. **Custom kernel evidence:** show the `tr_ring_fused_kernel` (and `tr_splitk_reduce_kernel`) rows with their grid and duration if available. Point out that the permute or copy kernels are gone.
5. **Supplementary evidence:**
   - Run the rank-16 trace separately: `--rank 16 --tokens 1 --output results/supp_rank16_t1_profile.json --profile-dir traces/rank16`.
   - Keep the §6 step-10 in-process profile of all five cases.
   - Include screenshots or Perfetto links if helpful.
6. **Optional deep dive:**
   - `nsys profile -t cuda,nvtx -o traces/nsys_rank8_t1 <python> benchmarks/benchmark.py ... --tokens 1 --output results/supp_nsys.json` gives a timeline including launch gaps.
   - `ncu --target-processes all -k regex:tr_ring --set full -o traces/ncu_r16_t32 ...` gives registers per thread, spills (local-memory traffic), achieved occupancy, L1/L2 hit rates and tensor-pipe utilization.
   - Performance counters may be disabled on cloud GPUs. If so, say so.

### 8.3 Supplementary runs (never mixed into the required tables)

- **Tuning sweep, in-process:** mimic the harness's event method (warm up, then 5 blocks × 20 calls between CUDA events) over the knobs in §5.8. Pick the configuration and freeze it into the §6 cell. Disclose that the tuning targeted these workloads on this GPU, and that this carries a risk of overfitting.
- **Ablations through the harness:**
  - Let `submission.py` read an optional environment variable (e.g. `TR_VARIANT=v1|v2`, `TR_SPLIT_K=<n>`) **only for these runs**.
  - Write them to `results/supp_ablation_<name>_rank{8,16}.json`.
  - Required runs use the defaults, and the notebook prints the environment right before them.
- **Repeat runs:** run each required command once or twice more into `results/supp_repeat<n>_*.json` and report the run-to-run spread.
- **Optional cold-L2 study (supports the system-implications section):**
  - Create N weight sets such that N × 10.55 MiB ≥ 4 × L2 (dense weights for the dense method, prepared operators for factorized).
  - Time calls that cycle through them, so that every call's weights have been evicted from L2.
  - Label it clearly as an emulation of a many-layer model, not a required result.
- **Optional CUDA-graph section:**
  - Capture all three methods the same way at a fixed T, and include the copy into the static input buffer inside the timed region.
  - Report capture time and graph memory-pool usage.
  - Keep it in its own section of the report (R5).

---

## 9. Writing `REPORT.md`

Copy `REPORT_TEMPLATE.md` and keep its headings. Keep the report concise and link to the notebook sections and artifacts. **Every number must come from a JSON file or trace in the submission.** No estimates may appear as results. Label analytic numbers (§4) as analytic.

| Template section | Content |
|---|---|
| 1. Findings | What was built (fused B→A→C Triton kernel with split-K). Headline comparison per case against reference and dense (stream medians). The honest verdict, e.g. "beats reference in X/5, loses to dense in Y/5 because…". Open uncertainties: warm-L2 microbenchmark, tuning on the same workload, run-to-run variance. |
| 2. Kernel and evidence | **Files and toolchain:** `tr_kernels.py::tr_ring_fused_kernel`, `tr_splitk_reduce_kernel`, Triton/torch/CUDA versions; adapted code: none, or cite source and license. **Reference bottleneck:** the §8.2 numbers. **Kernel:** the §5.1 paragraph, ownership (t, q, k-group), layouts, which data sits in registers and which in L1/L2, what is read and written, where it appears in the trace. **Preparation and state:** Bp, Cp and the A snapshot (bytes); per-call output and workspace; JIT cost cold and warm; remaining bottlenecks (padding efficiency, A re-reads, host launch floor). |
| 3. Correctness and results | Tests (pytest summary, §7 matrix), tolerances, observed errors against the reference's, harness changes (none). The 15-row table (5 cases × 3 methods), preparation/first-call, memory breakdown, speedups. Explain the unfavourable cases, and the trends across rank (ρ³ term, padding efficiency) and token count (launch-bound → throughput-bound, split-K). Optional graph results go in a separate subsection. |
| 4. System implications | See below. |
| 5. Reproduction and disclosure | Exact commands, including notebook execution; commit hash or ZIP SHA-256; `results/environment.txt`, GPU and driver; list of raw results and traces; GPU hours from the §6 step-17 log and "instance stopped at …"; AI disclosure: which tool, what it generated, and what the human verified independently (re-ran tests, read traces, checked the index algebra with the emulation). |

What to cover in section 4, System implications:
- **Larger batches and prefill:** factorized FLOPs grow as T·(29.5–197 MFLOP) against dense T·11 MFLOP, so dense wins in the compute-bound regime. Discuss switching strategy there: a cuBLAS-based path, or bounded per-tile weight reconstruction amortized over many tokens.
- **Larger rank:** the ρ³ stage-2 term grows 8× for every doubling of rank.
- **Full model:** weights of many layers do not stay in L2 (cite the cold-L2 study if you ran it), and freed memory can go to KV cache or batch size.
- **Real servers capture decode in CUDA graphs**, which removes host overhead, so GPU time becomes the fair metric.
- **Quality:** random factors say nothing about a TR-compressed pretrained model's accuracy.
- **Baselines to compare against:** INT8/INT4 weight-only GEMV, low-rank SVD and 2:4 sparsity, at matched quality and memory.
- **Integration work:** graph-capturability (no host syncs; fixed T per graph), per-layer shape specialization and compile cost, tensor-parallel sharding of ring factors, and end-to-end metrics (tokens/s, TTFT/TPOT percentiles, memory, energy).
- **What would change the approach:** for example, ncu showing L2-bound A re-reads (move to TT tiling or v2), host-bound T=1 (C++ launcher or graphs), or T ≥ ~64 (a different algorithm).

---

## 10. Discussion prep (the human must be able to answer these; put the answers in the notebook)

- Why B→A→C? Show the FLOP table and why the order does not change the result (associativity; only rounding differs).
- What does one program own, and why is there no race? Each `(t, q, k-group)` tile has exactly one writer. The split-K partials go to distinct workspace slots and are summed in a fixed order.
- What sits in registers, what in shared memory (tl.dot operand staging), and what in L1/L2? How many bytes does one program read per `k`? How often is each loaded `A_a` tile reused?
- What padding waste does each rank incur, and why accept it?
- Why SPLIT_K > 1 at T=1 but 1 at T=32? What does the second launch cost?
- What does CUDA-event stream time measure in this harness? It measures 20 calls back to back, so a host-bound method shows host dispatch rate, not kernel time. Why is it not kernel time?
- Why can dense still win despite 124× compression? (FLOPs, warm L2, cuBLAS GEMV quality, one launch.)
- Where does each memory number come from? Resident vs incremental vs reserved, and what the PyTorch counters miss (the Triton kernel cache and the CUDA context).
- How would you make it graph-capturable? How would you scale it to T=512 or to ρ=32?
- What would you measure before claiming a system-level win? (§9 section 4.)

---

## 11. Pitfalls checklist

- [ ] Padded lanes loaded without `other=0.0`, which contaminates dot products.
- [ ] A `tl.arange` size that is not a power of 2, or a `tl.dot` dimension below 16.
- [ ] An FP32 operand passed to `tl.dot`, which silently uses TF32.
- [ ] `T` passed as a kernel argument or constexpr (recompiles), or grid axes 1 or 2 that exceed 65,535.
- [ ] Calling a stale `factorized_inference.prepare_optimized` after `%%writefile` instead of the reloaded module.
- [ ] Using `!python` instead of `sys.executable`.
- [ ] Ablation environment variables left set during the required runs.
- [ ] GPU work running in the notebook while the harness measures.
- [ ] `results/` and `traces/` not added to git or the ZIP, because they are gitignored.
- [ ] A hidden `torch.cuda.synchronize()` or `.item()` in the forward.
- [ ] A global cache of prepared weights.
- [ ] Output-buffer reuse that is not documented.
- [ ] A dense-weight-sized allocation anywhere in the optimized path. Check this with the §6 step-10 allocation delta.
- [ ] Reporting kernel-only times as the headline instead of CUDA-event stream medians.
- [ ] Cherry-picking a single favourable run.
- [ ] Mixing supplementary or graph numbers into the required table.
- [ ] Hand-typed numbers in `REPORT.md`.
- [ ] Hiding JIT cost. Report first call with cold and warm Triton caches.
- [ ] Claiming that the tolerance pass means "percentage-accurate". It does not (IMPLEMENTATION.md).

---

## 12. Definition of done

- [ ] `solution.ipynb` runs top to bottom on the GPU with outputs saved, and it generates `tr_kernels.py` and `submission.py`.
- [ ] `pytest -q -rA`: all pass, 0 skipped on the GPU.
- [ ] The custom Triton kernel appears in the profile of **all five** required cases (notebook table) and in `traces/rank8/factorized_optimized_rank8_tokens1.json`.
- [ ] No dense weight is constructed. Persistent extra state is about 1× the factor bytes, and it is inventoried.
- [ ] `results/rank8.json`, `results/rank16.json`, `results/environment.txt` and `traces/rank8/` come from the exact required commands.
- [ ] `REPORT.md` has every template section, the 15-row table, preparation/first-call, the memory breakdown, errors, the profiler explanation, limitations, system implications, reproduction commands, GPU hours and the AI disclosure.
- [ ] Supplementary runs are labelled and kept separate. The optional graph section, if present, treats all methods the same way.
- [ ] Artifacts are force-added or zipped, and the revision identifier is recorded.
- [ ] The GPU instance is stopped, and this is stated in the report.
