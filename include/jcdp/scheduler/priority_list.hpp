/******************************************************************************
 * @file jcdp/scheduler/priority_list.hpp
 *
 * @brief This file is part of the JCDP package. It provides a priority list
 *        scheduler that uses the in-tree task dependencies of the elimination
 *        sequence to sort the operations. Given the sorted operation it
 *        performs a simple list scheduling. Results rather often in an optimal
 *        schedule.
 ******************************************************************************/

#ifndef JCDP_SCHEDULER_PRIORITY_LIST_HPP_
#define JCDP_SCHEDULER_PRIORITY_LIST_HPP_

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "jcdp/operation.hpp"
#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::scheduler {

class PriorityListScheduler : public Scheduler {
 public:
   virtual auto schedule_impl(
        Sequence& sequence, const std::size_t usable_threads, const std::size_t)
        -> std::size_t override final {

      std::vector<std::size_t> queue_cont(sequence.length());
      std::iota(queue_cont.begin(), queue_cont.end(), 0);

      std::priority_queue queue(
           [&sequence](const std::size_t& op_idx1, const std::size_t& op_idx2)
                -> bool {
              const std::size_t level_1 = sequence.level(op_idx1);
              const std::size_t level_2 = sequence.level(op_idx2);
              if (level_1 == level_2) {
                 return sequence.ops[op_idx1].fma < sequence.ops[op_idx2].fma;
              }
              return sequence.level(op_idx1) < sequence.level(op_idx2);
           },
           std::move(queue_cont));

      // Reset potential previous schedule
      for (Operation& op : sequence.ops) {
         op.is_scheduled = false;
      }

      std::vector<std::size_t> thread_loads(usable_threads, 0);
      while (!queue.empty()) {
         const std::size_t op_idx = queue.top();
         const std::size_t earliest_start = sequence.earliest_start(op_idx);

         Operation& op = sequence[op_idx];
         op.thread = 0;
         op.start_time = std::max(thread_loads[0], earliest_start);
         std::size_t current_idle_time = op.start_time - thread_loads[0];

         for (size_t t = 1; t < usable_threads; t++) {
            const std::size_t start_on_t = std::max(
                 thread_loads[t], earliest_start);
            const std::size_t idle_on_t = start_on_t - thread_loads[t];

            if (start_on_t < op.start_time) {
               op.thread = t;
               op.start_time = start_on_t;
               current_idle_time = idle_on_t;
            } else if (start_on_t == op.start_time) {
               if (idle_on_t < current_idle_time) {
                  op.thread = t;
                  op.start_time = start_on_t;
                  current_idle_time = idle_on_t;
               }
            }
         }

         thread_loads[op.thread] = op.start_time + op.fma;
         op.is_scheduled = true;
         queue.pop();
      }

      return sequence.makespan();
   }
};

}  // namespace jcdp::scheduler

#endif  // JCDP_SCHEDULER_PRIORITY_LIST_HPP_
