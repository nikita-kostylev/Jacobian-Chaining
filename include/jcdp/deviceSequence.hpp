#ifndef JCDP_DEVICE_SEQUENCE_HPP_
#define JCDP_DEVICE_SEQUENCE_HPP_

#include <cstddef>

#include "jcdp/operation.hpp"

namespace jcdp {

/* ========================= CONFIGURATION ========================= */

constexpr int MAX_SEQUENCE_LENGTH = 11; // FIXED FOR LIMITED TESTING. SHOULD BE ADJUSTABLE

/* ========================= DEVICE SEQUENCE ======================== */

struct DeviceSequence {
   Operation ops[MAX_SEQUENCE_LENGTH];
   std::size_t length;
   std::size_t best_makespan_output;
};

/* ========================= DEVICE FUNCTIONS ======================= */

// #pragma omp declare target

/* ------------------------- Makespan ------------------------------- */
/* thread == SIZE_MAX means "any thread"                              */

inline DeviceSequence device_make_max() {
   DeviceSequence seq {};
   seq.length = 1;

   Operation& op = seq.ops[0];

   op.fma = static_cast<std::size_t>(-1);  // SIZE_MAX without <limits>
   op.start_time = 0;
   op.thread = 0;
   op.is_scheduled = true;

   return seq;
}

inline std::size_t device_sequential_makespan(const DeviceSequence& seq) {
   std::size_t cost = 0;
   for (std::size_t i = 0; i < seq.length; ++i) {
      cost += seq.ops[i].fma;
   }
   return cost;
}

inline std::size_t makespan(
     const DeviceSequence& seq,
     std::size_t thread = static_cast<std::size_t>(-1)) {
   std::size_t cost = 0;

   for (std::size_t i = 0; i < seq.length; ++i) {
      const Operation& op = seq.ops[i];

      if (thread == static_cast<std::size_t>(-1) || op.thread == thread) {
         if (op.is_scheduled) {
            const std::size_t end = op.start_time + op.fma;
            if (end > cost)
               cost = end;
         }
      }
   }
   return cost;
}

inline std::size_t count_accumulations(const DeviceSequence& seq) {
   std::size_t count = 0;
   for (std::size_t i = 0; i < seq.length; ++i) {
      if (seq.ops[i].action == Action::ACCUMULATION) {
         ++count;
      }
   }
   return count;
}

/* ------------------ Sequential Makespan --------------------------- */

inline std::size_t sequential_makespan(const DeviceSequence& seq) {
   std::size_t sum = 0;
   for (std::size_t i = 0; i < seq.length; ++i) {
      sum += seq.ops[i].fma;
   }
   return sum;
}

/* ------------------ Is Scheduled ---------------------------------- */

inline bool is_scheduled(const DeviceSequence& seq) {
   for (std::size_t i = 0; i < seq.length; ++i) {
      if (!seq.ops[i].is_scheduled) {
         return false;
      }
   }
   return true;
}

/* ------------------ Is Schedulable -------------------------------- */

inline bool is_schedulable(const DeviceSequence& seq, std::size_t op_idx) {
   for (std::size_t i = 0; i < seq.length; ++i) {
      if (seq.ops[op_idx] < seq.ops[i]) {
         if (!seq.ops[i].is_scheduled) {
            return false;
         }
      }
   }
   return true;
}

/* ------------------ Earliest Start -------------------------------- */

inline std::size_t earliest_start(
     const DeviceSequence& seq, std::size_t op_idx) {
   std::size_t max_time = 0;

   for (std::size_t i = 0; i < seq.length; ++i) {
      if (seq.ops[op_idx] < seq.ops[i]) {
         const std::size_t end = seq.ops[i].start_time + seq.ops[i].fma;
         if (end > max_time) {
            max_time = end;
         }
      }
   }
   return max_time;
}

/* ------------------ Critical Path --------------------------------- */

inline std::size_t device_critical_path(const DeviceSequence& seq) {

   std::size_t max_cp = 0;

   for (std::size_t i = 0; i < seq.length; ++i) {

      std::size_t time = seq.ops[i].start_time + seq.ops[i].fma;

      std::size_t current = i;

      while (true) {
         bool found_parent = false;

         for (std::size_t j = 0; j < seq.length; ++j) {
            if (seq.ops[j] < seq.ops[current]) {
               const std::size_t parent_end = seq.ops[j].start_time +
                                              seq.ops[j].fma;

               if (parent_end > time) {
                  time = parent_end;
               }

               current = j;
               found_parent = true;
               break;
            }
         }

         if (!found_parent)
            break;
      }

      if (time > max_cp) {
         max_cp = time;
      }
   }

   return max_cp;
}

//#pragma omp end declare target

}  // namespace jcdp

template<>
struct std::formatter<jcdp::DeviceSequence> {
   template<class ParseContext>
   constexpr auto parse(ParseContext& ctx) {
      return ctx.begin();
   }

   template<class FormatContext>
   auto format(const jcdp::DeviceSequence& seq, FormatContext& ctx) {
      auto out = ctx.out();
      for (std::size_t i = 0; i < seq.length; ++i) {
         out = std::format_to(out, "{}\n", seq.ops[i]);
      }
      return out;
   }
};

#endif  // JCDP_DEVICE_SEQUENCE_HPP_