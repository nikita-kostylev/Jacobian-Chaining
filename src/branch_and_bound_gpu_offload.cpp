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
   std::array<std::size_t, 4> thread_loads_full_array {};
};

inline bool all_scheduled(const DeviceSequence& seq) {
   for (size_t i = 0; i < seq.length; ++i) {
      if (!seq.ops[i].is_scheduled)
         return false;
   }
   return true;
}

#pragma omp declare target

static DeviceSequence nonrecursive_schedule_op(
     std::size_t& best_makespan, DeviceSequence& working_copy,
     const std::size_t usable_threads, const std::size_t sequential_makespan) {
   std::array<std::size_t, 4> thread_loads {};
   thread_loads.fill(0);

   std::size_t makespan = 0;
   std::size_t idling_time = 0;

   Layer stack[16];
   std::size_t sp = 0;  // stack pointer = recursion depth

   DeviceSequence best_sequence = working_copy;

   // ---- initialize root frame ----
   stack[0] = Layer {};
   stack[0].op_idx = 0;
   stack[0].thread_idx = 0;
   stack[0].depth = 0;
   stack[0].thread_loads_full_array = thread_loads;

   // std::size_t count = 0;

   while (true) {
      std::size_t scheduled = 0;
      for (std::size_t i = 0; i < working_copy.length; ++i)
         if (working_copy.ops[i].is_scheduled)
            scheduled++;

      assert(scheduled == sp && "DFS depth != number of scheduled ops");

      // ===== FIND NEXT UNSCHEDULED SCHEDULABLE OP =====
      std::size_t op_idx = stack[sp].op_idx;
      while (op_idx < working_copy.length &&
             (working_copy.ops[op_idx].is_scheduled ||
              !is_schedulable(working_copy, op_idx))) {
         op_idx++;
      }

      // ===== LEAF NODE =====
      if (op_idx >= working_copy.length) {
         if (all_scheduled(working_copy)) {
            if (makespan < best_makespan) {
               best_makespan = makespan;
               for (std::size_t i = 0; i < working_copy.length; ++i) {
                  best_sequence.ops[i] = working_copy.ops[i];
                  best_sequence.ops[i].is_scheduled = true;
               }
               best_sequence.best_makespan_output = best_makespan;
            }
         }

         // BACKTRACK
         if (sp == 0)
            break;

         --sp;
         Layer& prev = stack[sp];

         const std::size_t old_op = prev.op_idx;
         working_copy.ops[old_op].is_scheduled = false;
         working_copy.ops[old_op].start_time = 0;

         thread_loads = prev.thread_loads_full_array;
         makespan = prev.makespan;
         idling_time = prev.idletime;

         stack[sp].thread_idx++;
         continue;
      }

      stack[sp].op_idx = op_idx;

      // ===== TRY THREADS =====
      if (stack[sp].thread_idx >= usable_threads) {
         if (sp == 0)
            break;

         --sp;
         Layer& prev = stack[sp];

         const std::size_t old_op = prev.op_idx;
         working_copy.ops[old_op].is_scheduled = false;
         working_copy.ops[old_op].start_time = 0;

         thread_loads = prev.thread_loads_full_array;
         makespan = prev.makespan;
         idling_time = prev.idletime;

         stack[sp].thread_idx++;
         continue;
      }

      const std::size_t t = stack[sp].thread_idx;

      // Schedule op
      assert(!working_copy.ops[op_idx].is_scheduled);
      assert(is_schedulable(working_copy, op_idx));

      working_copy.ops[op_idx].is_scheduled = true;
      const std::size_t est = std::max(
           thread_loads[t], earliest_start(working_copy, op_idx));

      working_copy.ops[op_idx].start_time = est;
      working_copy.ops[op_idx].thread = t;

      const std::size_t old_load = thread_loads[t];
      const std::size_t old_makespan = makespan;
      const std::size_t old_idle = idling_time;

      idling_time += (est - old_load);
      thread_loads[t] = est + working_copy.ops[op_idx].fma;
      makespan = std::max(makespan, thread_loads[t]);

      // LOWER BOUND
      const std::size_t lb = std::max(
           ((idling_time + sequential_makespan) / usable_threads),
           device_critical_path(working_copy));

      if (std::max(lb, makespan) < best_makespan) {
         // DESCEND (push frame)
         Layer next {};
         next.op_idx = 0;
         next.thread_idx = 0;
         next.depth = sp + 1;
         next.makespan = makespan;
         next.idletime = idling_time;
         next.thread_loads_full_array = thread_loads;

         assert(sp + 1 < 17 && "DFS stack overflow — infinite loop detected");
         stack[++sp] = next;
         continue;
      }

      // REVERT THREAD TRY
      working_copy.ops[op_idx].is_scheduled = false;
      working_copy.ops[op_idx].start_time = 0;
      thread_loads[t] = old_load;
      makespan = old_makespan;
      idling_time = old_idle;

      stack[sp].thread_idx++;
   }

   return best_sequence;
}

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
      op.thread = 0;
      op.start_time = 0;
   }

   const std::size_t lower_bound = sequence.critical_path();

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
      std::println("This sequence failed to run GPU");
      std::println("{}", sequence);
      std::println("{}", usable_threads);
      return 0;
   } else {
      best_makespan = result_sequence.best_makespan_output;
      for (size_t i = 0; i < sequence.length(); ++i) {
         sequence[i].thread = result_sequence.ops[i].thread;
         sequence[i].start_time = result_sequence.ops[i].start_time;
         sequence[i].is_scheduled = result_sequence.ops[i].is_scheduled;
      }
      // std::println("Usable threads for following sequence: {}",
      // usable_threads);
      // std::println("{}", result_sequence);
      /* if (all_scheduled(result_sequence)) {
         // std::println("All operations scheduled");
      } else {
         for (size_t i = 0; i < sequence.length(); i++) {
            std::println(
                 "operation {} is_scheduled: {}", i,
                 result_sequence.ops[i].is_scheduled);
         }
      } */

      return best_makespan;
   }
}

}  // namespace jcdp::scheduler