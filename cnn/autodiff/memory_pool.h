// Copyright 2018-2020. All Rights Reserved.
// Author: csukuangfj@gmail.com (Fangjun Kuang)

#pragma once

#include <memory>
#include <vector>

namespace cnn {

/** A memory arena for allocating space.
 *
 * @note Once space is allocated, it is NOT deallocated
 * until the arena dies.
 *
 * @tparam kBasicSize The basic unit for allocation. It is usually from
 * sizeof(SomeType).
 */
template <size_t kBasicSize>
class MemoryArena {
 public:
  /** Construct a memory arena with specified number of elements.
   *
   * It will allocate more space later on.
   *
   * @param[in] num_elements Value specifying the initial capacity
   * of the memory arena.
   */
  explicit MemoryArena(size_t num_elements) {
    size_t size_in_bytes = kBasicSize * num_elements;

    char* p = new char[size_in_bytes];
    buf_.emplace_back(p);
    base_ = p;
    end_ = base_ + size_in_bytes;
  }

  /** Copy constructor is disabled. */
  MemoryArena(const MemoryArena&) = delete;

  /** Assignment operator is disabled. */
  MemoryArena& operator=(const MemoryArena&) = delete;

  /** Move constructor is disabled. */
  MemoryArena(MemoryArena&&) = delete;

  /** Move assignment operator is disabled. */
  MemoryArena& operator=(MemoryArena&&) = delete;

  /** Allocate space for a given number of elements.
   *
   * @param [in] num_elements Number of elements to allocate.
   *
   * @return A pointer pointing to the allocated space.
   */
  void* Allocate(size_t num_elements) {
    size_t size_in_bytes = kBasicSize * num_elements;
    if (base_ + size_in_bytes >= end_) {
      // the remaining space is not large enough,
      // so we have to allocate new space.
      // To avoid extra number of allocations, we allocate
      // more space than required.
      char* p = new char[size_in_bytes * 8];
      buf_.emplace_back(p);
      base_ = p;
      end_ = base_ + size_in_bytes * 8;
      return p;
    }

    char* p = base_;
    base_ += size_in_bytes;
    return p;
  }

 private:
  std::vector<std::unique_ptr<char[]>> buf_;
  char* base_ = nullptr;
  char* end_ = nullptr;
};

template <size_t kBasicSize>
class MemoryPool {
 public:
  /** Construct a memory pool with specified number of elements.
   *
   * It will allocate more space later on.
   *
   * @param[in] num_elements It specifies the initial capacity
   * of the memory pool.
   */
  explicit MemoryPool(size_t num_elements) : arena_(num_elements) {}

  /** Copy constructor is disabled. */
  MemoryPool(const MemoryPool&) = delete;

  /** Assignment operator is disabled. */
  MemoryPool& operator=(const MemoryPool&) = delete;

  /** Move constructor is disabled. */
  MemoryPool(MemoryPool&&) = delete;

  /** Move assignment operator is disabled. */
  MemoryPool& operator=(MemoryPool&&) = delete;

  /** Allocate one element.
   *
   * @return A pointer pointing to the allocated space.
   */
  void* Allocate() {
    if (head) {
      auto* p = head;
      head = head->next;
      p->next = nullptr;
      return p;
    }

    auto* p = reinterpret_cast<Node*>(arena_.Allocate(1));
    p->next = nullptr;
    return p;
  }

  /** Free a given pointer.
   *
   * @param[in] p The pointer to free.
   */
  void Free(void* p) {
    if (p) {
      auto* q = reinterpret_cast<Node*>(p);
      q->next = head;
      head = q;
    }
  }

 private:
  /** @note Things become a little bit complicated if a union is used here. */
  struct Node {
    char buf[kBasicSize];
    Node* next;
  };

  MemoryArena<sizeof(Node)> arena_;  //!< memory arena for allocating memory.
  Node* head = nullptr;              //!< head of the free list.
};

}  // namespace cnn
