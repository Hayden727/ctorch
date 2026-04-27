//===- include/ctorch/intrusive_ptr.h - Minimal intrusive_ptr ---*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Minimal intrusive smart pointer used by Storage. The refcount lives next
/// to the data pointer in StorageImpl, so a Tensor handle stays in one cache
/// line. Modeled loosely on c10::intrusive_ptr but with only the operations
/// ctorch needs today.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_INTRUSIVE_PTR_H
#define CTORCH_INTRUSIVE_PTR_H

#include <atomic>
#include <cstdint>
#include <utility>

namespace ctorch {

/// Base class for any type that wants to be managed by intrusive_ptr.
/// Holds the atomic refcount inline.
class intrusive_ref_counted {
  public:
    intrusive_ref_counted() = default;
    intrusive_ref_counted(const intrusive_ref_counted&) = delete;
    intrusive_ref_counted& operator=(const intrusive_ref_counted&) = delete;
    intrusive_ref_counted(intrusive_ref_counted&&) = delete;
    intrusive_ref_counted& operator=(intrusive_ref_counted&&) = delete;

  protected:
    ~intrusive_ref_counted() = default;

  private:
    template <class U> friend class intrusive_ptr;
    mutable std::atomic<std::int64_t> refcount_{0};
};

/// Owning pointer to an `intrusive_ref_counted` derivative `T`.
/// Copy increments refcount; destructor / reset / move-from decrement and
/// `delete` on zero.
template <class T> class intrusive_ptr {
  public:
    intrusive_ptr() noexcept = default;

    explicit intrusive_ptr(T* p) noexcept : ptr_(p) { acquire(); }

    intrusive_ptr(const intrusive_ptr& other) noexcept : ptr_(other.ptr_) { acquire(); }

    intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    intrusive_ptr& operator=(const intrusive_ptr& other) noexcept {
        if (this != &other) {
            T* old = ptr_;
            ptr_ = other.ptr_;
            acquire();
            release_ptr(old);
        }
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& other) noexcept {
        if (this != &other) {
            T* old = ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            release_ptr(old);
        }
        return *this;
    }

    ~intrusive_ptr() { release_ptr(ptr_); }

    void reset() noexcept {
        T* old = ptr_;
        ptr_ = nullptr;
        release_ptr(old);
    }

    T* get() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    T* operator->() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    std::int64_t use_count() const noexcept {
        return ptr_ == nullptr ? 0 : ptr_->refcount_.load(std::memory_order_acquire);
    }

    template <class... Args> static intrusive_ptr<T> make(Args&&... args) {
        return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
    }

  private:
    void acquire() noexcept {
        if (ptr_ != nullptr) {
            ptr_->refcount_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static void release_ptr(T* p) noexcept {
        if (p == nullptr) {
            return;
        }
        if (p->refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete p;
        }
    }

    T* ptr_ = nullptr;
};

} // namespace ctorch

#endif // CTORCH_INTRUSIVE_PTR_H
