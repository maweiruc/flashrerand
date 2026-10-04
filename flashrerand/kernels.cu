// Experimental CUDA kernels. RNG layout is part of the replay contract.
// NVRTC installations need not expose host C standard-library headers.
// Keep the replay integer widths self-contained in the device translation unit.
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using int64_t = long long;
static_assert(sizeof(uint32_t) == 4, "Philox requires 32-bit uint32_t");
static_assert(sizeof(uint64_t) == 8, "Replay IDs require 64-bit uint64_t");
static_assert(sizeof(int64_t) == 8, "Assignment counts require 64-bit int64_t");

struct PhiloxWords { uint32_t x, y, z, w; };

__device__ __forceinline__ PhiloxWords philox(
    uint64_t seed, uint64_t candidate, uint32_t draw,
    uint32_t stage, uint32_t attempt) {
    PhiloxWords c = {uint32_t(candidate), uint32_t(candidate >> 32),
                     draw, (stage << 24) | attempt};
    uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        uint64_t p0 = uint64_t(0xD2511F53u) * c.x;
        uint64_t p1 = uint64_t(0xCD9E8D57u) * c.z;
        PhiloxWords next = {uint32_t(p1 >> 32) ^ c.y ^ k0,
                            uint32_t(p1), uint32_t(p0 >> 32) ^ c.w ^ k1,
                            uint32_t(p0)};
        c = next;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

__device__ __forceinline__ uint32_t bounded(
    uint64_t seed, uint64_t candidate, uint32_t draw, uint32_t stage,
    uint32_t bound, uint32_t* error) {
    uint32_t threshold = (uint32_t(0) - bound) % bound;
    uint32_t attempt = 0;
    while (true) {
        uint32_t raw = philox(seed, candidate, draw, stage, attempt).x;
        if (raw >= threshold) return raw % bound;
        if (attempt == 0xFFFFFFu) {
            atomicMax(error, 1u);  // Exhausted the replay address space.
            return 0;
        }
        ++attempt;
    }
}

extern "C" __global__ void generate_keys(
    const uint64_t* ids, uint64_t seed, int n, int count,
    const uint32_t* attempts, const unsigned char* active, uint64_t* keys) {
    uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t size = uint64_t(n) * count;
    for (; index < size; index += uint64_t(gridDim.x) * blockDim.x) {
        uint64_t b = index / n;
        if (!active[b]) continue;
        PhiloxWords r = philox(seed, ids[b], uint32_t(index % n), 1, attempts[b]);
        keys[index] = uint64_t(r.x) | (uint64_t(r.y) << 32);
    }
}

extern "C" __global__ void generate_sequential(
    const uint64_t* ids, uint64_t seed, int n, int n1, int count,
    unsigned char* w, uint32_t* error) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count) return;
    int remaining = n1;
    for (int i = 0; i < n; ++i) {
        int m = n - i;
        bool take = remaining == m;
        if (remaining != 0 && remaining != m)
            take = bounded(seed, ids[b], uint32_t(i), 2, uint32_t(m), error)
                   < uint32_t(remaining);
        w[uint64_t(b) * n + i] = take;
        remaining -= int(take);
    }
}

extern "C" __global__ void generate_fair_bits(
    const uint64_t* ids, uint64_t seed, int n, int count, unsigned char* w) {
    int lane = threadIdx.x & 31;
    int words_per_row = (n + 31) / 32;
    uint64_t word = uint64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    uint64_t size = uint64_t(words_per_row) * count;
    for (; word < size; word += uint64_t(gridDim.x) * (blockDim.x / 32)) {
        uint64_t b = word / words_per_row;
        uint32_t group = uint32_t(word % words_per_row);
        uint32_t bits = 0;
        if (lane == 0) bits = philox(seed, ids[b], group, 3, 0).x;
        bits = __shfl_sync(0xFFFFFFFFu, bits, 0);
        int i = int(group * 32) + lane;
        if (i < n) w[b * n + i] = (bits >> lane) & 1;
    }
}

extern "C" __global__ void repair_counts(
    const uint64_t* ids, uint64_t seed, int n, int n1, int count,
    const int64_t* counts, unsigned char* w, uint32_t* error) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count) return;
    int current = int(counts[b]);
    uint64_t trial = 0;
    while (current != n1) {
        if (trial >= (uint64_t(1) << 32)) {
            atomicMax(error, 2u);
            return;
        }
        uint32_t i = bounded(seed, ids[b], uint32_t(trial), 4, uint32_t(n), error);
        ++trial;
        unsigned char* value = w + uint64_t(b) * n + i;
        if (current > n1 && *value) { *value = 0; --current; }
        else if (current < n1 && !*value) { *value = 1; ++current; }
    }
}

// PK1: each thread writes one Philox word as at most four little-endian bytes.
// Byte stores are intentional: a packed row is not necessarily word-aligned.
#ifdef FLASHRERAND_PK1
extern "C" __global__ void generate_packed_fair_bits(
    const uint64_t* ids, uint64_t seed, int n, int count, unsigned char* packed) {
    int width = int((uint64_t(n) + 7) / 8);
    int words = int((uint64_t(n) + 31) / 32);
    uint64_t word = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t size = uint64_t(words) * count;
    for (; word < size; word += uint64_t(gridDim.x) * blockDim.x) {
        uint64_t b = word / words;
        uint32_t group = uint32_t(word % words);
        uint32_t bits = philox(seed, ids[b], group, 3, 0).x;
        for (int byte = 0; byte < 4; ++byte) {
            int j = int(group * 4) + byte;
            if (j >= width) break;
            uint32_t value = (bits >> (byte * 8)) & 0xFFu;
            int valid = n - j * 8;
            if (valid < 8) value &= (1u << valid) - 1u;
            packed[b * width + j] = (unsigned char)value;
        }
    }
}

extern "C" __global__ void count_packed_rows(
    const unsigned char* packed, int n, int count, int64_t* counts) {
    int width = int((uint64_t(n) + 7) / 8);
    uint64_t b = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; b < uint64_t(count); b += uint64_t(gridDim.x) * blockDim.x) {
        int current = 0;
        for (int j = 0; j < width; ++j)
            current += __popc(uint32_t(packed[b * width + j]));
        counts[b] = current;
    }
}

extern "C" __global__ void repair_packed_counts(
    const uint64_t* ids, uint64_t seed, int n, int n1, int count,
    const int64_t* counts, unsigned char* packed, uint32_t* error) {
    int width = int((uint64_t(n) + 7) / 8);
    uint64_t b = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; b < uint64_t(count); b += uint64_t(gridDim.x) * blockDim.x) {
        int current = int(counts[b]);
        uint64_t trial = 0;
        while (current != n1) {
            if (trial >= (uint64_t(1) << 32)) {
                atomicMax(error, 2u);
                return;
            }
            uint32_t i = bounded(seed, ids[b], uint32_t(trial), 4, uint32_t(n), error);
            ++trial;
            unsigned char* value = packed + b * width + i / 8;
            unsigned char mask = (unsigned char)(1u << (i % 8));
            bool take = (*value & mask) != 0;
            if (current > n1 && take) { *value &= (unsigned char)~mask; --current; }
            else if (current < n1 && !take) { *value |= mask; ++current; }
        }
    }
}
#endif

extern "C" __global__ void pack_rows(
    const unsigned char* w, unsigned char* packed, int n, int count) {
    uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int width = (n + 7) / 8;
    uint64_t size = uint64_t(width) * count;
    for (; index < size; index += uint64_t(gridDim.x) * blockDim.x) {
        uint64_t b = index / width;
        int start = int(index % width) * 8;
        unsigned char byte = 0;
        for (int bit = 0; bit < 8 && start + bit < n; ++bit)
            byte |= w[b * n + start + bit] << bit;
        packed[index] = byte;
    }
}

extern "C" __global__ void unpack_rows(
    const unsigned char* packed, unsigned char* w, int n, int count) {
    uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int width = (n + 7) / 8;
    uint64_t size = uint64_t(n) * count;
    for (; index < size; index += uint64_t(gridDim.x) * blockDim.x) {
        uint64_t b = index / n;
        int i = int(index % n);
        w[index] = (packed[b * width + i / 8] >> (i % 8)) & 1;
    }
}

template<typename T> __device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    return value;
}

// Four independent candidate warps. Each coordinate is completed over ALL
// subjects before its square enters the norm.
template<typename T> __global__ void score_reduce(
    const unsigned char* w, const T* z, const T* correction,
    int n, int p, int n1, int count, T scale, T* output) {
    int lane = threadIdx.x & 31;
    int b = blockIdx.x * 4 + threadIdx.x / 32;
    if (b >= count) return;
    T norm = 0;
    T pi = T(n1) / T(n);
    for (int j = lane; j < p; j += 32) {
        T value = 0;
        for (int i = 0; i < n; ++i)
            if (w[uint64_t(b) * n + i]) value += z[uint64_t(i) * p + j];
        value -= pi * correction[j];
        norm += value * value;
    }
    norm = warp_sum(norm);
    if (lane == 0) output[b] = scale * norm;
}

// Same tile layout/accumulation for dense and packed inputs; only W loading
// changes. Four candidates reuse each 32-subject by 32-coordinate Z tile.
template<typename T, bool Packed> __global__ void score_tiled(
    const unsigned char* w, const T* z, const T* correction,
    int n, int p, int n1, int count, T scale, T* output) {
    __shared__ T z_tile[32 * 32];
    __shared__ unsigned char w_tile[4 * 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x / 32;
    int b = blockIdx.x * 4 + warp;
    int width = Packed ? (n + 7) / 8 : n;
    T norm = 0;
    T pi = T(n1) / T(n);
    for (int j0 = 0; j0 < p; j0 += 32) {
        int j = j0 + lane;
        T value = 0;
        for (int i0 = 0; i0 < n; i0 += 32) {
            for (int index = threadIdx.x; index < 1024; index += 128) {
                int i = i0 + index / 32, col = j0 + index % 32;
                z_tile[index] = (i < n && col < p) ? z[uint64_t(i) * p + col] : T(0);
            }
            int i = i0 + lane;
            unsigned char bit = 0;
            if (b < count && i < n) {
                if (Packed) bit = (w[uint64_t(b) * width + i / 8] >> (i % 8)) & 1;
                else bit = w[uint64_t(b) * n + i];
            }
            w_tile[threadIdx.x] = bit;
            __syncthreads();
            for (int r = 0; r < 32; ++r)
                if (w_tile[warp * 32 + r]) value += z_tile[r * 32 + lane];
            __syncthreads();
        }
        if (j < p) {
            value -= pi * correction[j];
            norm += value * value;
        }
    }
    norm = warp_sum(norm);
    if (b < count && lane == 0) output[b] = scale * norm;
}

// PK0 bounded variants. Keep the baseline score_tiled symbol above untouched.
// Every candidate completes each coordinate across ALL subjects before its
// square enters the norm; candidate-tail warps still join every CTA barrier.
template<typename T, bool Packed, int Candidates, int Subjects>
__global__ void score_tiled_variant(
    const unsigned char* w, const T* z, const T* correction,
    int n, int p, int n1, int count, T scale, T* output) {
    static_assert(Candidates == 4 || Candidates == 8, "PK0 candidate tile");
    static_assert(Subjects == 32 || Subjects == 64, "PK0 subject tile");
    __shared__ T z_tile[Subjects * 32];
    __shared__ unsigned char w_tile[Candidates * Subjects];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x / 32;
    int b = blockIdx.x * Candidates + warp;
    int width = Packed ? (n + 7) / 8 : n;
    T norm = 0;
    T pi = T(n1) / T(n);
    for (int j0 = 0; j0 < p; j0 += 32) {
        int j = j0 + lane;
        T value = 0;
        // Use a wide tile base so rounding to a 64-subject tile cannot overflow
        // the signed subject bound supported by the original 32-subject path.
        for (uint64_t i0 = 0; i0 < uint64_t(n); i0 += Subjects) {
            for (int index = threadIdx.x; index < Subjects * 32;
                 index += Candidates * 32) {
                uint64_t i = i0 + index / 32;
                int col = j0 + index % 32;
                z_tile[index] = (i < uint64_t(n) && col < p)
                    ? z[i * p + col] : T(0);
            }
            for (int row = lane; row < Subjects; row += 32) {
                uint64_t i = i0 + row;
                unsigned char bit = 0;
                if (b < count && i < uint64_t(n)) {
                    if (Packed)
                        bit = (w[uint64_t(b) * width + i / 8] >> (i % 8)) & 1;
                    else bit = w[uint64_t(b) * n + i];
                }
                w_tile[warp * Subjects + row] = bit;
            }
            __syncthreads();
            for (int row = 0; row < Subjects; ++row)
                if (w_tile[warp * Subjects + row])
                    value += z_tile[row * 32 + lane];
            __syncthreads();
        }
        if (j < p) {
            value -= pi * correction[j];
            norm += value * value;
        }
    }
    norm = warp_sum(norm);
    if (b < count && lane == 0) output[b] = scale * norm;
}

#ifdef FLASHRERAND_REPAIR_FUSED
// Only the private exhaustion tests override these compile-time bounds.
#ifndef FLASHRERAND_FUSED_ATTEMPT_MAX
#define FLASHRERAND_FUSED_ATTEMPT_MAX 0xFFFFFFu
#endif
#ifndef FLASHRERAND_FUSED_TRIAL_LIMIT
#define FLASHRERAND_FUSED_TRIAL_LIMIT (uint64_t(1) << 32)
#endif
#ifndef FLASHRERAND_FUSED_FORCE_REJECT
#define FLASHRERAND_FUSED_FORCE_REJECT 0
#endif

// Unlike the legacy helper, failure is returned explicitly so the candidate
// stops repairing without abandoning a block barrier or using a fallback draw.
__device__ __forceinline__ bool fused_bounded_checked(
    uint64_t seed, uint64_t candidate, uint32_t draw, uint32_t bound,
    uint32_t* error, uint32_t& output) {
    uint32_t threshold = (uint32_t(0) - bound) % bound;
    uint32_t attempt = 0;
    while (true) {
        uint32_t raw = philox(seed, candidate, draw, 4, attempt).x;
        if (!FLASHRERAND_FUSED_FORCE_REJECT && raw >= threshold) {
            output = raw % bound;
            return true;
        }
        if (attempt == FLASHRERAND_FUSED_ATTEMPT_MAX) {
            atomicMax(error, 1u);
            return false;
        }
        ++attempt;
    }
}

// Repair-first prototype: the complete candidate state lives in shared memory.
// Eight candidate warps reuse one 64x32 Z tile, matching C1's score order.
// Production debug_packed is null; only scores and the sticky flag are written.
extern "C" __global__ void score_fused_repair(
    const uint64_t* ids, uint64_t seed, const double* z,
    const double* correction, int n, int p, int n1, int count,
    double scale, double* output, uint32_t* error,
    unsigned char* debug_packed) {
    __shared__ uint32_t membership[8][32];
    __shared__ double z_tile[64 * 32];
    __shared__ unsigned char w_tile[8 * 64];
    __shared__ int failed[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x / 32;
    int b = blockIdx.x * 8 + warp;
    bool active = b < count;
    uint32_t bits = 0;
    int valid = n - lane * 32;
    if (active && valid > 0) {
        bits = philox(seed, ids[b], uint32_t(lane), 3, 0).x;
        if (valid < 32) bits &= (1u << valid) - 1u;
    }
    membership[warp][lane] = bits;
    int current = warp_sum(int(__popc(bits)));
    __syncwarp();
    if (lane == 0) {
        failed[warp] = 0;
        if (active) {
            uint64_t trial = 0;
            while (current != n1) {
                if (trial >= FLASHRERAND_FUSED_TRIAL_LIMIT) {
                    atomicMax(error, 2u);
                    failed[warp] = 1;
                    break;
                }
                uint32_t i;
                if (!fused_bounded_checked(seed, ids[b], uint32_t(trial),
                                           uint32_t(n), error, i)) {
                    failed[warp] = 1;
                    break;
                }
                ++trial;
                uint32_t mask = 1u << (i % 32);
                uint32_t& word = membership[warp][i / 32];
                bool take = (word & mask) != 0;
                if (current > n1 && take) { word &= ~mask; --current; }
                else if (current < n1 && !take) { word |= mask; ++current; }
            }
        }
    }
    // All candidate tails and failed candidates continue through every barrier.
    __syncthreads();
    if (debug_packed && active) {
        int width = (n + 7) / 8;
        for (int byte = lane; byte < width; byte += 32)
            debug_packed[uint64_t(b) * width + byte] =
                (unsigned char)(membership[warp][byte / 4] >> (8 * (byte % 4)));
    }
    double value = 0;
    for (int i0 = 0; i0 < n; i0 += 64) {
        for (int index = threadIdx.x; index < 64 * 32; index += 256) {
            int i = i0 + index / 32, col = index % 32;
            z_tile[index] = (i < n && col < p) ? z[uint64_t(i) * p + col] : 0.0;
        }
        for (int row = lane; row < 64; row += 32) {
            int i = i0 + row;
            unsigned char bit = 0;
            if (active && !failed[warp] && i < n)
                bit = (membership[warp][i / 32] >> (i % 32)) & 1;
            w_tile[warp * 64 + row] = bit;
        }
        __syncthreads();
        for (int row = 0; row < 64; ++row)
            if (w_tile[warp * 64 + row]) value += z_tile[row * 32 + lane];
        __syncthreads();
    }
    double norm = 0;
    if (lane < p) {
        value -= (double(n1) / double(n)) * correction[lane];
        norm += value * value;
    }
    norm = warp_sum(norm);
    if (active && lane == 0) output[b] = failed[warp] ? 0.0 : scale * norm;
}
#endif

template<typename T, int MaxP> __global__ void score_fused_sequential(
    const uint64_t* ids, uint64_t seed, const T* z, const T* correction,
    int n, int p, int n1, int count, T scale, T* output, uint32_t* error) {
    int lane = threadIdx.x & 31;
    int b = blockIdx.x * 4 + threadIdx.x / 32;
    if (b >= count) return;
    T values[MaxP / 32];
    #pragma unroll
    for (int slot = 0; slot < MaxP / 32; ++slot) values[slot] = 0;
    int remaining = n1;
    for (int i = 0; i < n; ++i) {
        int take = 0;
        if (lane == 0) {
            int m = n - i;
            take = remaining == m;
            if (remaining != 0 && remaining != m)
                take = bounded(seed, ids[b], uint32_t(i), 2, uint32_t(m), error)
                       < uint32_t(remaining);
            remaining -= take;
        }
        take = __shfl_sync(0xFFFFFFFFu, take, 0);
        if (take) {
            #pragma unroll
            for (int slot = 0; slot < MaxP / 32; ++slot) {
                int j = lane + 32 * slot;
                if (j < p) values[slot] += z[uint64_t(i) * p + j];
            }
        }
    }
    T norm = 0;
    T pi = T(n1) / T(n);
    #pragma unroll
    for (int slot = 0; slot < MaxP / 32; ++slot) {
        int j = lane + 32 * slot;
        if (j < p) {
            T value = values[slot] - pi * correction[j];
            norm += value * value;
        }
    }
    norm = warp_sum(norm);
    if (lane == 0) output[b] = scale * norm;
}
