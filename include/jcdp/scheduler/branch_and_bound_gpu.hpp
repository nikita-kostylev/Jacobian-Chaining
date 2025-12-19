/******************************************************************************
 * @file jcdp/scheduler/branch_and_bound_gpu.hpp
 *
 * @brief GPU-flavoured copy of the Branch & Bound scheduler. Implementation is
 *        currently identical to the CPU version; only the type name differs so
 *        it can be wired separately.
 ******************************************************************************/

#ifndef JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_
#define JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <algorithm>
#include <cstddef>
#include <print>
#include <vector>

#include "jcdp/operation.hpp"
#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::scheduler {

class BranchAndBoundSchedulerGPU : public Scheduler {
 public:
   virtual auto schedule_impl(
        Sequence& sequence, const std::size_t usable_threads,
        const std::size_t upper_bound) -> std::size_t override final {
      const std::size_t sequential_makespan = sequence.sequential_makespan();

      Sequence working_copy = sequence;
      std::size_t best_makespan = upper_bound;

      std::vector<std::size_t> thread_loads(usable_threads, 0);
      std::size_t makespan = 0;
      std::size_t idling_time = 0;

      // Reset potential previous schedule
      for (Operation& op : working_copy) {
         op.is_scheduled = false;
      }

      const std::size_t lower_bound = working_copy.critical_path();

      if (lower_bound >= upper_bound) {
         return lower_bound;
      }


      
      // Helper: generate all k-combinations (indices) from n items (0..n-1).
      auto generate_combinations = [](std::size_t n, std::size_t k)
           -> std::vector<std::vector<std::size_t>> {
         std::vector<std::vector<std::size_t>> res;
         if (k > n) return res;
         if (k == 0) {
            res.emplace_back();
            return res;
         }
         std::vector<std::size_t> comb(k);
         for (std::size_t i = 0; i < k; ++i) comb[i] = i;
         while (true) {
            res.push_back(comb);
            long long i = static_cast<long long>(k) - 1;
            while (i >= 0 && comb[static_cast<std::size_t>(i)] ==
                                 static_cast<std::size_t>(i) + n - k) {
               --i;
            }
            if (i < 0) break;
            ++comb[static_cast<std::size_t>(i)];
            for (std::size_t j = static_cast<std::size_t>(i) + 1; j < k; ++j)
               comb[j] = comb[j - 1] + 1;
         }
         return res;
      };





      
      // --- recursion: branching/search logic starts here ---
      auto schedule_op = [&](auto& schedule_next_op) -> bool {
         // Return if time's up
         if (!remaining_time()) {
            return true;
         }

         bool everything_scheduled = true;
         for (std::size_t op_idx = 0; op_idx < sequence.length(); ++op_idx) {
            if (working_copy[op_idx].is_scheduled) {
               continue;
            }
            everything_scheduled = false;

            if (!working_copy.is_schedulable(op_idx)) {
               continue;
            }

            working_copy[op_idx].is_scheduled = true;
            bool tried_empty_processor = false;
            const std::size_t start = working_copy.earliest_start(op_idx);

            for (size_t t = 0; t < usable_threads; t++) {
               // We only need to check one empty processor (w.l.o.g.)
               if (thread_loads[t] == 0) {
                  if (tried_empty_processor) {
                     break;
                  }
                  tried_empty_processor = true;
               }

               const std::size_t old_start_time =
                    working_copy[op_idx].start_time;
               const std::size_t start_time = std::max(thread_loads[t], start);
               working_copy[op_idx].start_time = start_time;

               const std::size_t old_thread_load = thread_loads[t];
               thread_loads[t] = start_time + sequence[op_idx].fma;

               const std::size_t old_idling_time = idling_time;
               idling_time += (start_time - old_thread_load);

               const std::size_t old_makespan = makespan;
               makespan = std::max(makespan, thread_loads[t]);

               const std::size_t lb = std::max(
                    ((idling_time + sequential_makespan) / usable_threads),
                    working_copy.critical_path());
               if (std::max(lb, makespan) < best_makespan) {
                  working_copy[op_idx].thread = t;

                  // Perform branching and exit if lower bound is reached
                  if (schedule_next_op(schedule_next_op)) {
                     return true;
                  }
               }

               thread_loads[t] = old_thread_load;
               idling_time = old_idling_time;
               makespan = old_makespan;
               working_copy[op_idx].start_time = old_start_time;
            }

            working_copy[op_idx].is_scheduled = false;
         }

         if (everything_scheduled) {
            if (makespan < best_makespan) {
               best_makespan = makespan;
               for (size_t i = 0; i < sequence.length(); ++i) {
                  sequence[i].thread = working_copy[i].thread;
                  sequence[i].start_time = working_copy[i].start_time;
                  sequence[i].is_scheduled = true;
               }
               if (best_makespan <= lower_bound) {
                  return true;
               }
            }
         }

         return false;
      };

      schedule_op(schedule_op);
      return best_makespan;
   }
};

}  // namespace jcdp::scheduler

#endif  // JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_
