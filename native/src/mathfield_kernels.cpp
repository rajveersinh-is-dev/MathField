#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>

#ifdef _WIN32
  #define MATHFIELD_API __declspec(dllexport)
#else
  #define MATHFIELD_API __attribute__((visibility("default")))
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

template <typename T>
inline T mathfield_clamp(T val, T lo, T hi) {
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

template <typename T>
inline T safe_pow_val(T u, T v) {
    if (std::isnan(u) || std::isnan(v)) return (T)0.0;
    if (std::fabs(u) < (T)1e-15 && std::fabs(v) < (T)1e-15) return (T)1.0;
    T abs_u = (u >= (T)0.0) ? u : -u;
    if (abs_u < (T)1e-15) abs_u = (T)1e-15;
    T clamped_v = mathfield_clamp(v, (T)-80.0, (T)80.0);

    T s = (T)1.0;
    if (u < (T)0.0) {
        T r = std::round(clamped_v);
        if (std::fabs(clamped_v - r) < (T)1e-7) {
            long long int_v = (long long)std::fabs(r);
            if (int_v % 2 != 0) {
                s = (T)-1.0;
            }
        }
    }
    T log_abs_u = std::log(abs_u);
    T exp_val = mathfield_clamp(clamped_v * log_abs_u, (T)-80.0, (T)80.0);
    return s * std::exp(exp_val);
}

extern "C" {

// OpenMP thread query & configuration
MATHFIELD_API int get_openmp_thread_count() {
    int n = 1;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}

MATHFIELD_API void set_openmp_thread_count(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

// Residual computation: res[i] = pred[i] - target[i]
MATHFIELD_API void compute_residuals_f64(
    const double* pred,
    const double* target,
    double* res,
    size_t n
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        res[i] = pred[i] - target[i];
    }
}

// Sum of squared errors: total = sum((pred[i] - target[i])^2)
MATHFIELD_API double compute_sse_f64(
    const double* pred,
    const double* target,
    size_t n
) {
    double total = 0.0;
    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double diff = pred[i] - target[i];
        total += diff * diff;
    }
    return total;
}

// Weighted residual calculation
MATHFIELD_API void compute_weighted_residuals_f64(
    const double* pred,
    const double* target,
    const double* weights,
    double* res,
    size_t n
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        res[i] = (pred[i] - target[i]) * weights[i];
    }
}

// 2D Wendland C2 RBF Field Evaluation
MATHFIELD_API void evaluate_wendland_rbf2d_f64(
    const double* cx,
    const double* cy,
    const double* weights,
    const double* epsilons,
    size_t m_centers,
    const double* x,
    const double* y,
    size_t n_points,
    double* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n_points; ++i) {
        double xi = x[i];
        double yi = y[i];
        double sum = 0.0;
        for (size_t j = 0; j < m_centers; ++j) {
            double dx = xi - cx[j];
            double dy = yi - cy[j];
            double r = std::sqrt(dx * dx + dy * dy) * epsilons[j];
            if (r < 1.0) {
                double t = 1.0 - r;
                double t4 = t * t * t * t;
                sum += weights[j] * t4 * (4.0 * r + 1.0);
            }
        }
        out[i] = sum;
    }
}

// Bytecode Virtual Machine for Dynamic AST Expressions
MATHFIELD_API int evaluate_bytecode_f64(
    const uint32_t* opcodes,
    const int32_t* operands,
    size_t n_instructions,
    const double* const_pool,
    size_t n_consts,
    const double* x,
    const double* y,
    size_t n_points,
    const double* params,
    size_t n_params,
    double* out
) {
    int error_code = 0;
    #pragma omp parallel
    {
        double stack[256];
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)n_points; ++i) {
            if (error_code != 0) continue;
            double xi = x[i];
            double yi = y[i];
            int sp = 0;

            for (size_t k = 0; k < n_instructions; ++k) {
                uint32_t op = opcodes[k];
                int32_t arg = operands[k];

                switch (op) {
                    case 1:  if (sp >= 256) { error_code = -1; goto finish; } stack[sp++] = xi; break;                   // LOAD_X
                    case 2:  if (sp >= 256) { error_code = -1; goto finish; } stack[sp++] = yi; break;                   // LOAD_Y
                    case 3:  if (sp >= 256) { error_code = -1; goto finish; } stack[sp++] = (arg < (int32_t)n_consts) ? const_pool[arg] : 0.0; break; // LOAD_CONST
                    case 4:  if (sp >= 256) { error_code = -1; goto finish; } stack[sp++] = (arg < (int32_t)n_params) ? params[arg] : 0.0; break;     // LOAD_PARAM
                    case 5:  { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; stack[sp++] = a + b; break; } // ADD
                    case 6:  { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; stack[sp++] = a - b; break; } // SUB
                    case 7:  { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; stack[sp++] = a * b; break; } // MUL
                    case 8:  { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; // DIV
                               double s = (b >= 0.0) ? 1.0 : -1.0;
                               stack[sp++] = a / (b + 1e-12 * s); break; }
                    case 9:  { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; // POW
                               stack[sp++] = safe_pow_val(a, b); break; }
                    case 10: { if (sp < 2) { error_code = -2; goto finish; } double b = stack[--sp]; double a = stack[--sp]; // MOD
                               stack[sp++] = std::fmod(a, std::max(std::fabs(b), 1e-12)); break; }
                    case 11: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = -stack[sp-1]; break;         // NEG
                    case 13: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::sin(stack[sp-1]); break; // SIN
                    case 14: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::cos(stack[sp-1]); break; // COS
                    case 15: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(std::tan(stack[sp-1]), -1e6, 1e6); stack[sp-1] = v; break; } // TAN
                    case 16: { if (sp < 1) { error_code = -2; goto finish; } double v = stack[sp-1]; // SINC
                               if (std::fabs(v) < 1e-12) stack[sp-1] = 1.0;
                               else { double pv = M_PI * v; stack[sp-1] = std::sin(pv) / pv; }
                               break; }
                    case 17: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -1.0, 1.0); stack[sp-1] = std::asin(v); break; } // ASIN
                    case 18: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -1.0, 1.0); stack[sp-1] = std::acos(v); break; } // ACOS
                    case 19: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::atan(stack[sp-1]); break; // ATAN
                    case 20: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -80.0, 80.0); stack[sp-1] = std::exp(v); break; } // EXP
                    case 21: { if (sp < 1) { error_code = -2; goto finish; } double v = std::max(stack[sp-1], 1e-15); stack[sp-1] = std::log(v); break; } // LOG
                    case 22: { if (sp < 1) { error_code = -2; goto finish; } double raw = stack[sp-1]; double v = (std::isnan(raw) || raw < 0.0) ? 0.0 : raw; stack[sp-1] = std::sqrt(v); break; } // SQRT
                    case 23: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::cbrt(stack[sp-1]); break; // CBRT
                    case 24: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = stack[sp-1] * stack[sp-1]; break; // SQUARE
                    case 25: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::fabs(stack[sp-1]); break; // ABS
                    case 26: { if (sp < 1) { error_code = -2; goto finish; } double v = stack[sp-1]; stack[sp-1] = (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0); break; } // SIGN
                    case 27: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -80.0, 80.0); stack[sp-1] = 1.0 / (1.0 + std::exp(-v)); break; } // SIGMOID
                    case 28: { if (sp < 2) { error_code = -2; goto finish; } double r_rad = stack[--sp]; double r_dist = stack[--sp]; // RBF_WENDLAND
                               double safe_rad = std::max(std::fabs(r_rad), 1e-12);
                               double norm_r = std::max(r_dist, 0.0) / safe_rad;
                               if (norm_r >= 1.0) { stack[sp++] = 0.0; }
                               else { double t = 1.0 - norm_r; double t4 = t * t * t * t; stack[sp++] = t4 * (4.0 * norm_r + 1.0); }
                               break; }
                    case 29: { if (sp < 2) { error_code = -2; goto finish; } double eps = stack[--sp]; double r = stack[--sp]; // RBF_GAUSS
                               double sc = eps * r; double sc2 = std::min(sc * sc, 80.0);
                               stack[sp++] = std::exp(-sc2); break; }
                    case 30: { if (sp < 2) { error_code = -2; goto finish; } double eps = stack[--sp]; double r = stack[--sp]; // RBF_MQ
                               double sc = mathfield_clamp(eps * r, -1e80, 1e80); double sc2 = std::min(sc * sc, 1e160); stack[sp++] = std::sqrt(1.0 + sc2); break; }
                    case 31: { if (sp < 1) { error_code = -2; goto finish; } double v = stack[sp-1]; double av = std::fabs(v); double mv = std::max(v, 0.0); // SOFTPLUS
                               stack[sp-1] = std::log1p(std::exp(-av)) + mv; break; }
                    case 32: { if (sp < 3) { error_code = -2; goto finish; } double hi = stack[--sp]; double lo = stack[--sp]; double val = stack[--sp]; // CLAMP
                               stack[sp++] = mathfield_clamp(val, lo, hi); break; }
                    case 33: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -80.0, 80.0); stack[sp-1] = std::sinh(v); break; } // SINH
                    case 34: { if (sp < 1) { error_code = -2; goto finish; } double v = mathfield_clamp(stack[sp-1], -80.0, 80.0); stack[sp-1] = std::cosh(v); break; } // COSH
                    case 35: if (sp < 1) { error_code = -2; goto finish; } stack[sp-1] = std::tanh(stack[sp-1]); break; // TANH
                    case 36: goto finish; // RET
                }
            }
        finish:
            out[i] = (sp > 0) ? stack[sp-1] : 0.0;
        }
    }
    return error_code;
}

} // extern "C"
