// Lumen Runtime Library - Tensor Operations
// C++ header for tensor runtime

#ifndef LUMEN_TENSOR_HPP
#define LUMEN_TENSOR_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>

namespace lumen {

enum class DType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    I64,
};

class Tensor {
public:
    void* data;
    std::vector<int64_t> shape;
    DType dtype;
    int64_t ndim;
    int64_t size;
    bool requires_grad;

    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::F32)
        : shape(shape), dtype(dtype), requires_grad(false) {
        ndim = shape.size();
        size = 1;
        for (auto s : shape) {
            size *= s;
        }

        size_t bytes = size * dtype_size(dtype);
        data = malloc(bytes);
        memset(data, 0, bytes);
    }

    ~Tensor() {
        if (data) {
            free(data);
        }
    }

    // Get element size for dtype
    static size_t dtype_size(DType dtype) {
        switch (dtype) {
            case DType::F32: return 4;
            case DType::F16: return 2;
            case DType::BF16: return 2;
            case DType::I8: return 1;
            case DType::I32: return 4;
            case DType::I64: return 8;
            default: return 4;
        }
    }

    // Access data as float*
    float* data_ptr() {
        return static_cast<float*>(data);
    }

    // Fill with zeros
    void zeros() {
        memset(data, 0, size * dtype_size(dtype));
    }

    // Fill with value
    void fill(float value) {
        float* ptr = data_ptr();
        for (int64_t i = 0; i < size; i++) {
            ptr[i] = value;
        }
    }

    // Xavier initialization
    void xavier(int64_t fan_in, int64_t fan_out) {
        float std = sqrtf(2.0f / (fan_in + fan_out));
        float* ptr = data_ptr();
        for (int64_t i = 0; i < size; i++) {
            // Simple random (not cryptographically secure)
            ptr[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * std;
        }
    }
};

// Matrix multiplication: C = A @ B
Tensor* matmul(Tensor* A, Tensor* B) {
    // Assume 2D matrices for simplicity
    int64_t M = A->shape[0];
    int64_t K = A->shape[1];
    int64_t N = B->shape[1];

    auto C = new Tensor({M, N}, A->dtype);

    float* a_data = A->data_ptr();
    float* b_data = B->data_ptr();
    float* c_data = C->data_ptr();

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }

    return C;
}

// Element-wise ReLU
Tensor* relu(Tensor* input) {
    auto output = new Tensor(input->shape, input->dtype);

    float* in_data = input->data_ptr();
    float* out_data = output->data_ptr();

    for (int64_t i = 0; i < input->size; i++) {
        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;
    }

    return output;
}

// Element-wise Sigmoid
Tensor* sigmoid(Tensor* input) {
    auto output = new Tensor(input->shape, input->dtype);

    float* in_data = input->data_ptr();
    float* out_data = output->data_ptr();

    for (int64_t i = 0; i < input->size; i++) {
        out_data[i] = 1.0f / (1.0f + expf(-in_data[i]));
    }

    return output;
}

// Element-wise Tanh
Tensor* tanh_op(Tensor* input) {
    auto output = new Tensor(input->shape, input->dtype);

    float* in_data = input->data_ptr();
    float* out_data = output->data_ptr();

    for (int64_t i = 0; i < input->size; i++) {
        out_data[i] = tanhf(in_data[i]);
    }

    return output;
}

// Element-wise addition
Tensor* add(Tensor* A, Tensor* B) {
    auto C = new Tensor(A->shape, A->dtype);

    float* a_data = A->data_ptr();
    float* b_data = B->data_ptr();
    float* c_data = C->data_ptr();

    for (int64_t i = 0; i < A->size; i++) {
        c_data[i] = a_data[i] + b_data[i];
    }

    return C;
}

// Element-wise multiplication
Tensor* mul(Tensor* A, Tensor* B) {
    auto C = new Tensor(A->shape, A->dtype);

    float* a_data = A->data_ptr();
    float* b_data = B->data_ptr();
    float* c_data = C->data_ptr();

    for (int64_t i = 0; i < A->size; i++) {
        c_data[i] = a_data[i] * b_data[i];
    }

    return C;
}

} // namespace lumen

#endif // LUMEN_TENSOR_HPP
