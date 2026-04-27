//===- include/ctorch/storage.h - Reference-counted byte buffer -*- C++ -*-===//
//
// Part of the ctorch Project, under the MIT License.
// See LICENSE for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage is the device-tagged byte buffer backing one or more Tensors.
/// Multiple Tensors may alias the same Storage; the buffer is freed back to
/// its allocator when the last alias drops.
///
//===----------------------------------------------------------------------===//

#ifndef CTORCH_STORAGE_H
#define CTORCH_STORAGE_H

#include "ctorch/allocator.h"
#include "ctorch/device.h"
#include "ctorch/intrusive_ptr.h"

#include <cstddef>

namespace ctorch {

namespace detail {

class StorageImpl : public intrusive_ref_counted {
  public:
    StorageImpl(std::size_t nbytes, Device device, Allocator* allocator);
    ~StorageImpl();

    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) = delete;
    StorageImpl& operator=(StorageImpl&&) = delete;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t nbytes() const noexcept { return nbytes_; }
    Device device() const noexcept { return device_; }
    Allocator* allocator() const noexcept { return allocator_; }

  private:
    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    Device device_{};
    Allocator* allocator_ = nullptr;
};

} // namespace detail

/// Reference-counted handle to a byte buffer. Copying is cheap (one atomic
/// increment); the buffer is released to the underlying Allocator when the
/// last handle drops.
class Storage {
  public:
    /// Allocates \p nbytes via \p allocator and zero-initializes the buffer.
    /// Passing `allocator == nullptr` falls back to `default_allocator(d.kind)`.
    Storage(std::size_t nbytes, Device d, Allocator* allocator = nullptr);

    Storage() = default;

    void* data() { return impl_ ? impl_->data() : nullptr; }
    const void* data() const { return impl_ ? impl_->data() : nullptr; }
    std::size_t nbytes() const { return impl_ ? impl_->nbytes() : 0; }
    Device device() const { return impl_ ? impl_->device() : Device::cpu(); }
    Allocator* allocator() const { return impl_ ? impl_->allocator() : nullptr; }

    /// Number of live `Storage` handles referencing this buffer.
    std::int64_t use_count() const { return impl_.use_count(); }

    bool defined() const { return static_cast<bool>(impl_); }

  private:
    intrusive_ptr<detail::StorageImpl> impl_;
};

} // namespace ctorch

#endif // CTORCH_STORAGE_H
