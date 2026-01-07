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

#include <stack>
#include <filesystem>
#include <iostream>
#include <memory>

#include "jcdp/operation.hpp"
#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::scheduler {

struct Layer{
  // loop cursors / current choice
  size_t  op_idx = 0;      // where to continue scanning ops
  size_t  thread_idx  = 0;      // next thread to try for this op
  size_t depth = 0;           // depth in the search tree

  time_t start_time_op = 0;
  time_t load_on_thread  = 0;
  time_t idletime  = 0;
  size_t makespan = 0;

  std::vector<size_t> thread_loads_full;
};

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

      bool addInitAcc = false;

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

      Sequence base_sequence = working_copy;

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

      auto place_initial_accumulations = [&](const std::vector<std::size_t>& early_accs) -> bool {
         // Return if time's up
         if (!remaining_time()) {
            return false;
         }
         for (size_t t = 0; t < std::min(early_accs.size(),usable_threads); t++) {
            working_copy[early_accs[t]].is_scheduled = true;
            working_copy[early_accs[t]].start_time = 0;
            working_copy[early_accs[t]].thread = t;

            thread_loads[t] = sequence[early_accs[t]].fma;
            makespan = std::max(makespan, thread_loads[t]);
         }

         return true;
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

      auto nonrecursive_schedule_op = [&]() -> bool {
         std::stack<Layer> progress_stack;
         bool revert_op_idx = false;
         bool revert_thread_idx = false;
         bool skip_changes = false;
         std::size_t op_idx = 0;
         std::size_t thread_idx = 0;
         std::size_t depth = 0;
         //std::size_t old_thread_load = 0;

         for(std::size_t op_idx_temp = 0; op_idx_temp < sequence.length(); ++op_idx_temp){
            if (working_copy[op_idx_temp].is_scheduled) {
               depth++;
            }
         }



         Layer initial_layer;
         initial_layer.op_idx = 0;
         initial_layer.thread_idx = 0;    
         initial_layer.depth = depth;
         initial_layer.start_time_op = working_copy[0].start_time;
         initial_layer.load_on_thread = thread_loads[0];
         initial_layer.makespan = makespan;
         initial_layer.idletime = idling_time;
         initial_layer.thread_loads_full = thread_loads;
         progress_stack.push(initial_layer);

         //check whether optimal tree?
         std::vector<Mode> modes_in_sequence = {Mode::TANGENT,Mode::TANGENT,Mode::ADJOINT,Mode::ADJOINT,Mode::TANGENT,Mode::TANGENT,Mode::ADJOINT,Mode::ADJOINT,Mode::TANGENT,Mode::TANGENT,Mode::NONE,Mode::TANGENT,Mode::TANGENT,Mode::ADJOINT,Mode::TANGENT,Mode::TANGENT,Mode::ADJOINT,Mode::NONE,Mode::NONE,Mode::NONE};
         std::vector<Action> actions_in_sequence = {Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::MULTIPLICATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::ELIMINATION,Action::MULTIPLICATION,Action::MULTIPLICATION,Action::MULTIPLICATION};

         bool optimal_tree = true;
         for (size_t i = 0; i < sequence.length(); i++){
            if(sequence[i].mode!=modes_in_sequence[i] || sequence[i].action!=actions_in_sequence[i]){
               optimal_tree = false;
               break;            
            }
         }

         


         while(remaining_time() > 0.0){  //And some other metric when tree finished

            



            skip_changes = false;

            if(optimal_tree && depth == 8 && false){
               std::println("SKIPPED;  At depth {}, op idx {}, thread idx {}, with opx.isScheduled {}", depth, op_idx, thread_idx,working_copy[op_idx].is_scheduled);
            }
            //Skip already scheduled or unschedulable ops, revert if at end
            if (working_copy[op_idx].is_scheduled or !working_copy.is_schedulable(op_idx)) {
               op_idx++;
               if(op_idx >= sequence.length()){
                  revert_op_idx = true;
                  skip_changes = true;
               }else{
                  continue;
               }
            }

            if(optimal_tree && depth == 8){
               ////std::println("Depth: {}, Op idx: {}, Thread idx: {}, Best makespan: {}, Current makespan: {}", depth, op_idx, thread_idx, best_makespan, makespan);
            }
               //Calculate current value changes
            if(!skip_changes){
               //old_thread_load = thread_loads[thread_idx];
               working_copy[op_idx].is_scheduled = true;
               const std::size_t start_time = std::max(thread_loads[thread_idx], working_copy.earliest_start(op_idx));
               working_copy[op_idx].start_time = start_time;
               idling_time += (start_time - thread_loads[thread_idx]);
               thread_loads[thread_idx] = start_time + sequence[op_idx].fma;
               makespan = std::max(makespan, thread_loads[thread_idx]);
               if(working_copy[op_idx].action == Action::ACCUMULATION){
                  //std::println("Placed {} {} at {}, {} to {}  fma {}  makespan {}",working_copy[op_idx].action,working_copy[op_idx].mode,working_copy[op_idx].i,working_copy[op_idx].k+1,working_copy[op_idx].j+1,  working_copy[op_idx].fma, makespan);                  
               }    
               //std::println("Placed {} {} at thread {}, depth {}, makespan {}",working_copy[op_idx].action,working_copy[op_idx].mode,thread_idx,depth,makespan);           
            }

            //If finished schedule, save if best yet and revert to previous state
            if (depth >= sequence.length() - 1) {
               ////std::println("Finished a schedule with makespan {}", makespan);
               if (makespan < best_makespan) {
                  best_makespan = makespan;
                  for (size_t i = 0; i < sequence.length(); ++i) {
                     sequence[i].thread = working_copy[i].thread;
                     sequence[i].start_time = working_copy[i].start_time;
                     sequence[i].is_scheduled = true;
                  }
               }
               revert_thread_idx = true;
            }  

            //If lowed bound still good, go deeper, else revert
            const std::size_t lb = std::max(((idling_time + sequential_makespan) / usable_threads),working_copy.critical_path());
            ////std::println("  LB: {}, critpath   {}", lb, working_copy.critical_path());
            if (std::max(lb, makespan) < best_makespan) {
               working_copy[op_idx].thread = thread_idx;
               Layer current_layer;
               current_layer.op_idx = op_idx;
               current_layer.thread_idx = thread_idx;
               current_layer.depth = depth++;
               current_layer.start_time_op = working_copy[op_idx].start_time;
               //current_layer.load_on_thread = thread_loads[thread_idx];
               current_layer.makespan = makespan;
               current_layer.idletime = idling_time;
               current_layer.thread_loads_full = thread_loads;   //full array restore
               progress_stack.push(current_layer);
               op_idx=0;
               thread_idx=0; 
            }else{
               if(optimal_tree && depth == 10){
                  std::println("Pruned");
               }
               revert_thread_idx = true;
            }

            if(revert_thread_idx){
               revert_thread_idx = false;

               Layer previous_state = progress_stack.top();

               working_copy[op_idx].is_scheduled = false;
               working_copy[op_idx].start_time = previous_state.start_time_op;
               //thread_loads[thread_idx] = old_thread_load;
               makespan = previous_state.makespan;
               idling_time = previous_state.idletime;

               thread_idx = thread_idx + 1;
               if(thread_idx >= usable_threads){
                  revert_op_idx = true;
               }
               thread_loads = previous_state.thread_loads_full; //full array restore
               if(optimal_tree && depth == 8){
                  ////std::println("threadload {} ", thread_loads);
               }
            }

            if(revert_op_idx){
               revert_op_idx = false;
               if(depth == 0){
                  return true; // Finished entire search
               }
               depth--;
               Layer previous_state = progress_stack.top();
               progress_stack.pop();
               working_copy[op_idx].start_time = 0;
               op_idx = previous_state.op_idx;  // mabye +1? im not sure
               working_copy[op_idx].is_scheduled = false;
               thread_idx = 0;
               //working_copy[op_idx].start_time = previous_state.start_time_op;   probably useless
               //thread_loads[thread_idx] = previous_state.load_on_thread;
               makespan = previous_state.makespan;
               idling_time = previous_state.idletime;
               thread_loads = progress_stack.top().thread_loads_full;  //full array restore
               if(op_idx >= sequence.length()){
                  std::println("reached max opidx");
               }
            }
         }
         return false;
      };


      if(addInitAcc){
         if (accumulation_indices.size() < usable_threads) {
            schedule_op(schedule_op);
            return best_makespan;
         }

         const auto initial_combinations = generate_combinations(accumulation_indices.size(), usable_threads);

         bool explored = false;
         for (const auto& combination : initial_combinations) {
            if(!remaining_time()){
               break;
            }

            std::println("{:.1f}", remaining_time());
            working_copy = base_sequence;
            std::fill(thread_loads.begin(), thread_loads.end(), 0);
            makespan = 0;
            idling_time = 0;

            std::vector<std::size_t> ops_to_place;
            ops_to_place.reserve(combination.size());
            for (std::size_t idx : combination) {
               ops_to_place.push_back(accumulation_indices[idx]);
            }

            if (!place_initial_accumulations(ops_to_place)) {
               continue;
            }

            explored = true;
            schedule_op(schedule_op);
         }

         if (!explored) {
            schedule_op(schedule_op);
         }
         return best_makespan;
      } else {
         addInitAcc = nonrecursive_schedule_op();  
         return best_makespan;
      }
   }
};

}  // namespace jcdp::scheduler

#endif  // JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_