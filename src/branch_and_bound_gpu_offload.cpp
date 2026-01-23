#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>

#include "jcdp/deviceSequence.hpp"
#include "jcdp/operation.hpp"
#include "jcdp/scheduler/branch_and_bound_gpu.hpp"
#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::scheduler {

struct Layer {

   size_t op_idx = 0;
   size_t next_op_idx = 0;
   size_t thread_idx = 0;
   size_t depth = 0;
   time_t start_time_op = 0;
   time_t idletime = 0;
   size_t makespan = 0;

   std::array<std::size_t, 20> thread_loads_full_array {};  // Value has to be
                                                            // fixed for GPU.
                                                            // Selected smaller
                                                            // value, to reduce
                                                            // size.
};

#pragma omp declare target

static DeviceSequence nonrecursive_schedule_op(
     std::size_t& best_makespan, DeviceSequence& working_copy,
     const std::size_t usable_threads, const std::size_t sequential_makespan) {

   std::array<std::size_t, 20> thread_loads {};  // Value has to be fixed for
                                                 // GPU. Selected smaller value,
                                                 // to reduce size.
   thread_loads.fill(0);

   std::size_t makespan = 0;
   std::size_t idling_time = 0;

   Layer stack_array[20];  // Value has to be fixed for GPU. Selected smaller
                           // value, to reduce size.
   std::size_t stack_pointer = 0;
   bool revert_depth = false;
   bool revert_op_idx = false;
   bool revert_thread_idx = false;
   bool skip_changes = false;
   std::size_t op_idx = 0;
   std::size_t thread_idx = 0;
   std::size_t depth = 0;
   DeviceSequence sequence = working_copy;

   for (std::size_t op_idx_temp = 0; op_idx_temp < working_copy.length;
        ++op_idx_temp) {
      if (working_copy.ops[op_idx_temp].is_scheduled) {
         depth++;
      }
   }

   // create initial layer
   Layer initial_layer;
   initial_layer.op_idx = 0;
   initial_layer.next_op_idx = 0;
   initial_layer.thread_idx = 0;
   initial_layer.depth = depth;
   initial_layer.start_time_op = working_copy.ops[0].start_time;
   initial_layer.makespan = makespan;
   initial_layer.idletime = idling_time;
   initial_layer.thread_loads_full_array = thread_loads;
   stack_array[stack_pointer++] = initial_layer;

   int timer_replacement = 0;
   while (timer_replacement < 10000) {  // Rudementary replacement for timer.
                                        // Could be changed into either input
                                        // value to binary or some estimation
                                        // value(some mapping from dp duration
                                        // to gpu iterations)
      timer_replacement++;

      if (op_idx >= working_copy.length && depth == 0) {
         return sequence;
      }

      if (op_idx >= working_copy.length) {
         op_idx = working_copy.length - 1;
      }

      skip_changes = false;

      // Find the next non scheduled and schedulable operation
      if (working_copy.ops[op_idx].is_scheduled or
          !is_schedulable(working_copy, op_idx)) {
         op_idx++;
         stack_array[stack_pointer - 1].next_op_idx = op_idx;
         if (op_idx >= sequence.length) {
            revert_op_idx = true;
            skip_changes = true;
         } else {
            continue;
         }
      }

      if (thread_idx >= usable_threads) {
         revert_thread_idx = true;
         skip_changes = true;
      }

      // Schedule the selected operation on the selected thread
      if (!skip_changes) {
         working_copy.ops[op_idx].is_scheduled = true;
         const std::size_t start_time = std::max(
              thread_loads[thread_idx], earliest_start(working_copy, op_idx));
         working_copy.ops[op_idx].start_time = start_time;
         idling_time += (start_time - thread_loads[thread_idx]);
         thread_loads[thread_idx] = start_time + working_copy.ops[op_idx].fma;
         makespan = std::max(makespan, thread_loads[thread_idx]);
      }

      // Reached a leaf node, update best_makestpan if necessary
      if (depth >= working_copy.length - 1) {
         if (makespan < best_makespan) {
            best_makespan = makespan;
            for (size_t i = 0; i < working_copy.length; ++i) {
               sequence.ops[i].thread = working_copy.ops[i].thread;
               sequence.ops[i].start_time = working_copy.ops[i].start_time;
               sequence.ops[i].is_scheduled = true;
            }
            sequence.best_makespan_output = best_makespan;
         }
         revert_thread_idx = true;
      }

      // Check against lower bound and go deeper if possible
      if (!skip_changes && depth < working_copy.length - 1) {
         const std::size_t lb = std::max(
              ((idling_time + sequential_makespan) / usable_threads),
              device_critical_path(working_copy));
         if (std::max(lb, makespan) < best_makespan) {
            working_copy.ops[op_idx].thread = thread_idx;
            Layer current_layer;
            current_layer.op_idx = op_idx;
            current_layer.next_op_idx = 0;
            current_layer.thread_idx = thread_idx;
            current_layer.depth = depth++;
            current_layer.start_time_op = working_copy.ops[op_idx].start_time;
            current_layer.makespan = makespan;
            current_layer.idletime = idling_time;
            current_layer.thread_loads_full_array = thread_loads;
            stack_array[stack_pointer++] = current_layer;
            op_idx = 0;
            thread_idx = 0;
         } else {
            revert_thread_idx = true;
         }
      }

      // Revert the current changes and prepare for next iteration to try with
      // next thread
      if (revert_thread_idx) {
         revert_thread_idx = false;
         Layer previous_state = stack_array[stack_pointer - 1];
         if (op_idx < working_copy.length) {
            working_copy.ops[op_idx].start_time = 0;
            working_copy.ops[op_idx].is_scheduled = false;
         }
         makespan = previous_state.makespan;
         idling_time = previous_state.idletime;
         thread_idx = thread_idx + 1;
         // if thread overflow, then next operation should be scheduled
         if (thread_idx >= usable_threads) {
            revert_op_idx = true;
         }
         thread_loads = previous_state.thread_loads_full_array;
      }

      // Revert the current changes and prepare for next iteration to try with
      // next operation
      if (revert_op_idx) {
         revert_op_idx = false;
         Layer& previous_state = stack_array[stack_pointer - 1];
         if (op_idx < working_copy.length) {
            working_copy.ops[op_idx].start_time = 0;
            working_copy.ops[op_idx].is_scheduled = false;
            previous_state.next_op_idx = previous_state.next_op_idx + 1;
         }
         op_idx = previous_state.next_op_idx;
         thread_idx = 0;
         makespan = previous_state.makespan;
         idling_time = previous_state.idletime;
         if (stack_pointer > 0) {
            thread_loads =
                 stack_array[stack_pointer - 1].thread_loads_full_array;
         }
         // if operation overflow, then revert one level up in the search tree
         if (op_idx >= working_copy.length) {
            revert_depth = true;
         }
      }

      // Revert the operation one level up in the search tree and continue with
      // next thread
      if (revert_depth) {
         revert_depth = false;
         // if depth is zero, then we are done with all possibilities
         if (depth == 0) {
            return sequence;
         }
         depth--;
         size_t old_idx = stack_array[stack_pointer - 1].op_idx;
         working_copy.ops[old_idx].is_scheduled = false;
         working_copy.ops[old_idx].start_time = 0;
         stack_pointer--;
         Layer previous_state = stack_array[stack_pointer - 1];
         op_idx = previous_state.op_idx;
         thread_idx = previous_state.thread_idx + 1;
         makespan = previous_state.makespan;
         idling_time = previous_state.idletime;
         thread_loads = previous_state.thread_loads_full_array;
      }
   }
   return sequence;
};

#pragma omp end declare target

auto BranchAndBoundSchedulerGPU::schedule_impl(
     Sequence& sequence, const std::size_t usable_threads,
     const std::size_t upper_bound) -> std::size_t {

   std::size_t sequential_makespan = sequence.sequential_makespan();

   Sequence working_copy = sequence;
   std::size_t best_makespan = upper_bound;

   // Reset potential previous schedule
   for (Operation& op : working_copy) {
      op.is_scheduled = false;
   }

   const std::size_t lower_bound = working_copy.critical_path();

   if (lower_bound >= upper_bound) {
      return lower_bound;
   }

   std::vector<std::size_t> accumulation_indices;
   accumulation_indices.reserve(sequence.length());
   for (std::size_t idx = 0; idx < sequence.length(); ++idx) {
      if (sequence[idx].action == Action::ACCUMULATION) {
         accumulation_indices.push_back(idx);
      }
   }

   // Change to gpu compatible version of Sequence
   DeviceSequence result_sequence;
   DeviceSequence device_working_copy;

   for (std::size_t i = 0; i < working_copy.length(); ++i) {
      device_working_copy.ops[i] = working_copy[i];
   }
   device_working_copy.length = working_copy.length();

   // run code on GPU
   bool notrangpu = false;
#pragma omp target map(                                                        \
          to : best_makespan, device_working_copy, usable_threads,             \
               sequential_makespan) map(from : result_sequence)                \
     map(tofrom : notrangpu)
   {

      notrangpu = !omp_is_initial_device();
      if (notrangpu) {
         result_sequence = nonrecursive_schedule_op(
              best_makespan, device_working_copy, usable_threads,
              sequential_makespan);
      }
   }

   // Return gpu output with catch if gpu offload failed
   if (!notrangpu) {
      return 0;
   } else {
      best_makespan = result_sequence.best_makespan_output;
      /*       for (size_t i = 0; i < sequence.length(); ++i) {
               sequence[i].thread = result_sequence.ops[i].thread;
               sequence[i].start_time = result_sequence.ops[i].start_time;
               sequence[i].is_scheduled = true;
            }
       */
      return best_makespan;
   }
}

}  // namespace jcdp::scheduler