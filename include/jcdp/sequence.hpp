/******************************************************************************
 * @file jcdp/sequence.hpp
 *
 * @brief GPU-friendly Sequence with fixed-size operation array.
 ******************************************************************************/

#ifndef JCDP_SEQUENCE_HPP_
#define JCDP_SEQUENCE_HPP_

#include <cassert>
#include <cstddef>
#include <limits>

#include "jcdp/operation.hpp"

namespace jcdp {

// 🔧 Compile-time capacity (adjust as needed)
constexpr std::size_t MAX_SEQUENCE_LENGTH = 1024;

class Sequence {
 public:
   // ===== Data =====
   Operation ops[MAX_SEQUENCE_LENGTH];
   std::size_t len = 0;

   std::size_t best_makespan_output = 0;

   // ===== Constructors =====
   Sequence() = default;

   explicit Sequence(const Operation& op) {
      push_back(op);
   }

   // ===== Basic access =====
   inline std::size_t length() const {
      return len;
   }

   inline Operation& operator[](std::size_t i) {
      assert(i < len);
      return ops[i];
   }

   inline const Operation& operator[](std::size_t i) const {
      assert(i < len);
      return ops[i];
   }

   // ===== Modifiers =====
   inline void push_back(const Operation& op) {
      assert(len < MAX_SEQUENCE_LENGTH);
      ops[len++] = op;
   }

   inline void pop_back() {
      assert(len > 0);
      --len;
   }

   inline Operation& back() noexcept {
      assert(len > 0);
      return ops[len - 1];
   }

   inline const Operation& back() const noexcept {
      assert(len > 0);
      return ops[len - 1];
   }

   inline auto operator+=(const Operation& rhs) -> Sequence& {
      push_back(rhs);
      return *this;
   }

   inline auto operator+=(const Sequence& rhs) -> Sequence& {
      for (std::size_t i = 0; i < rhs.length(); ++i) {
         push_back(rhs.ops[i]);
      }
      return *this;
   }

   inline auto operator+(const Sequence& rhs) -> const Sequence {
      Sequence res = *this;
      res += rhs;
      return res;
   }

   inline auto operator+(const Operation& rhs) -> const Sequence {
      Sequence res = *this;
      res += rhs;
      return res;
   }

   inline void clear() {
      len = 0;
   }

   // ===== Scheduling utilities =====
   inline std::size_t makespan(
        std::size_t thread = static_cast<std::size_t>(-1)) const {
      std::size_t cost = 0;
      for (std::size_t i = 0; i < len; ++i) {
         const Operation& op = ops[i];
         if (thread == static_cast<std::size_t>(-1) || op.thread == thread) {
            assert(op.is_scheduled);
            const std::size_t end = op.start_time + op.fma;
            if (end > cost)
               cost = end;
         }
      }
      return cost;
   }

   inline std::size_t sequential_makespan() const {
      std::size_t sum = 0;
      for (std::size_t i = 0; i < len; ++i) {
         sum += ops[i].fma;
      }
      return sum;
   }

   inline bool is_scheduled() const {
      for (std::size_t i = 0; i < len; ++i) {
         if (!ops[i].is_scheduled)
            return false;
      }
      return true;
   }

   inline bool is_schedulable(std::size_t op_idx) const {
      assert(op_idx < len);
      for (std::size_t i = 0; i < len; ++i) {
         if (ops[op_idx] < ops[i]) {
            if (!ops[i].is_scheduled)
               return false;
         }
      }
      return true;
   }

   inline std::size_t earliest_start(std::size_t op_idx) const {
      assert(op_idx < len);
      std::size_t max_time = 0;
      for (std::size_t i = 0; i < len; ++i) {
         if (ops[op_idx] < ops[i]) {
            const std::size_t t = ops[i].start_time + ops[i].fma;
            if (t > max_time)
               max_time = t;
         }
      }
      return max_time;
   }

   inline std::size_t count_accumulations() const {
      std::size_t count = 0;
      for (std::size_t i = 0; i < len; ++i) {
         if (ops[i].action == Action::ACCUMULATION)
            ++count;
      }
      return count;
   }

   // ===== Critical path =====
   inline std::size_t parent(std::size_t op_idx) const {
      assert(op_idx < len);
      for (std::size_t i = 0; i < len; ++i) {
         if (ops[i] < ops[op_idx])
            return i;
      }
      return static_cast<std::size_t>(-1);
   }

   inline std::size_t critical_path(
        std::size_t op_idx, std::size_t start_time = 0) const {
      start_time = (start_time > ops[op_idx].start_time) ?
                        start_time :
                        ops[op_idx].start_time;

      const std::size_t end = start_time + ops[op_idx].fma;
      const std::size_t p = parent(op_idx);
      if (p != static_cast<std::size_t>(-1)) {
         return critical_path(p, end);
      }
      return end;
   }

   inline std::size_t critical_path() const {
      std::size_t max_cp = 0;
      for (std::size_t i = 0; i < len; ++i) {
         const std::size_t cp = critical_path(i);
         if (cp > max_cp)
            max_cp = cp;
      }
      return max_cp;
   }

   // ===== Factory =====
   inline static Sequence make_max() {
      Sequence s;
      Operation op {};
      op.fma = std::numeric_limits<std::size_t>::max();
      op.is_scheduled = true;
      s.push_back(op);
      return s;
   }
};

}  // namespace jcdp

// ---- formatter (CPU only) ----
template<>
struct std::formatter<jcdp::Sequence> {
   constexpr auto parse(auto& ctx) {
      return ctx.begin();
   }

   auto format(const jcdp::Sequence& seq, auto& ctx) const {
      auto out = ctx.out();
      for (std::size_t i = 0; i < seq.length(); ++i) {
         out = std::format_to(out, "{}\n", seq[i]);
      }
      return out;
   }
};

#endif  // JCDP_SEQUENCE_HPP_
