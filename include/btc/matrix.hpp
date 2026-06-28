#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace btc {

// Dense row-major Boolean matrix backed by uint8_t.
// Keeps values normalised to 0/1; bitwise OR on raw bytes gives correct results.
class BoolMatrix {
public:
    BoolMatrix() : rows_(0), cols_(0) {}

    BoolMatrix(std::size_t rows, std::size_t cols, bool init = false)
        : rows_(rows), cols_(cols),
          data_(rows * cols, static_cast<uint8_t>(init ? 1 : 0)) {}

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    bool get(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j] != 0;
    }

    void set(std::size_t i, std::size_t j, bool v) noexcept {
        data_[i * cols_ + j] = static_cast<uint8_t>(v ? 1 : 0);
    }

    // Element-wise OR in-place
    BoolMatrix& operator|=(const BoolMatrix& rhs) {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument("BoolMatrix::operator|=: size mismatch");
        for (std::size_t k = 0; k < data_.size(); ++k)
            data_[k] |= rhs.data_[k];
        return *this;
    }

    BoolMatrix operator|(const BoolMatrix& rhs) const {
        BoolMatrix result(*this);
        result |= rhs;
        return result;
    }

    bool operator==(const BoolMatrix& rhs) const noexcept {
        return rows_ == rhs.rows_ && cols_ == rhs.cols_ && data_ == rhs.data_;
    }

    bool operator!=(const BoolMatrix& rhs) const noexcept {
        return !(*this == rhs);
    }

    // Raw pointer to row i — exposed for cache-friendly matmul loops.
    const uint8_t* row_data(std::size_t i) const noexcept {
        return data_.data() + i * cols_;
    }
    uint8_t* row_data(std::size_t i) noexcept {
        return data_.data() + i * cols_;
    }

    static BoolMatrix identity(std::size_t n) {
        BoolMatrix m(n, n, false);
        for (std::size_t i = 0; i < n; ++i)
            m.set(i, i, true);
        return m;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<uint8_t> data_;
};

} // namespace btc
