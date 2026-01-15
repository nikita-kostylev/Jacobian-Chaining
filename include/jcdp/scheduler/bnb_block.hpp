/******************************************************************************
 * @file jcdp/scheduler/branch_and_bound.hpp
 *
 * @brief This file is part of the JCDP package. It provides branch & bound
 *        algorithm to find the optimal schedule for a given elimination
 *        sequence.
 ******************************************************************************/

#ifndef JCDP_SCHEDULER_BRANCH_AND_BOUND_BLOCK_HPP_
#define JCDP_SCHEDULER_BRANCH_AND_BOUND_BLOCK_HPP_

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

class BnBBlockScheduler {
 public:
   BnBBlockScheduler() = default;
   ~BnBBlockScheduler() = default;

   std::size_t schedule(
        Sequence& sequence, const std::size_t threads,
        const std::size_t upper_bound = std::numeric_limits<std::size_t>::max()) {

      // We can never use more threads than we have accumulations
      std::size_t usable_threads = sequence.count_accumulations();
      if (threads > 0 && threads < usable_threads) {
         usable_threads = threads;
      }

      return schedule_impl(sequence, usable_threads, upper_bound);
   }
   std::size_t schedule_impl(
        Sequence& sequence, const std::size_t usable_threads,
        const std::size_t upper_bound) {
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

      auto schedule_op = [&](auto& schedule_next_op) -> bool {
         // Return if time's up
         //if (!remaining_time()) {
         //   return true;
         //}

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

#endif  // JCDP_SCHEDULER_BRANCH_AND_BOUND_BLOCK_HPP_
