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
  size_t  next_op_idx = 0; // next op to schedule
  size_t  thread_idx  = 0;      // next thread to try for this op
  size_t  depth = 0;           // depth in the search tree

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

      auto nonrecursive_schedule_op = [&]() -> bool {
         std::stack<Layer> progress_stack;
         bool revert_depth = false;
         bool revert_op_idx = false;
         bool revert_thread_idx = false;
         bool skip_changes = false;
         std::size_t op_idx = 0;
         std::size_t thread_idx = 0;
         std::size_t depth = 0;
         std::size_t debug8count = 0;

         for(std::size_t op_idx_temp = 0; op_idx_temp < sequence.length(); ++op_idx_temp){
            if (working_copy[op_idx_temp].is_scheduled) {
               depth++;
            }
         }



         Layer initial_layer;
         initial_layer.op_idx = 0;
         initial_layer.next_op_idx = 0;
         initial_layer.thread_idx = 0;    
         initial_layer.depth = depth;
         initial_layer.start_time_op = working_copy[0].start_time;
         initial_layer.load_on_thread = thread_loads[0];
         initial_layer.makespan = makespan;
         initial_layer.idletime = idling_time;
         initial_layer.thread_loads_full = thread_loads;
         progress_stack.push(initial_layer);

         std::vector<Mode> modes_in_sequence = {Mode::ADJOINT,Mode::TANGENT,Mode::ADJOINT,Mode::ADJOINT,Mode::TANGENT,Mode::ADJOINT,Mode::TANGENT,Mode::NONE,Mode::NONE,Mode::NONE,Mode::NONE};
         std::vector<Action> actions_in_sequence = {Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ACCUMULATION,Action::ELIMINATION,Action::ELIMINATION,Action::MULTIPLICATION,Action::MULTIPLICATION,Action::MULTIPLICATION,Action::MULTIPLICATION};

         bool optimal_tree = false;   // set to true for debugging the loop with setup in text
         for (size_t i = 0; i < sequence.length(); i++){
            if(sequence[i].mode!=modes_in_sequence[i] || sequence[i].action!=actions_in_sequence[i]){
               optimal_tree = false;
               break;            
            }
         }

         if (sequence.length() == 11){
            for (size_t i = 0; i < sequence.length(); i++){
               ////std::println("{} , {}", sequence[i].mode, sequence[i].action);  enable for finding the current tree/sequence
            }
         }
         

         std::println("{}----------------------------------------------------------------------------{}",sequence.length(),usable_threads);

         while(remaining_time() > 0.0){

            std::vector<size_t> target{8722552u, 11946528u, 7683592u, 9603792u};

            if(thread_loads == target){
               optimal_tree = false;
               ////std::println("Double the threadloads");  use for debugging the loop with setup in text
            }

            if(op_idx >= sequence.length() && depth == 0){
               return true;
            }

            if(op_idx >= sequence.length()){
               std::println("opidx {} exceeded length {}", op_idx, sequence.length());
               op_idx = sequence.length() -1;
            }

            skip_changes = false;

            if (working_copy[op_idx].is_scheduled or !working_copy.is_schedulable(op_idx)) {
               op_idx++;
               progress_stack.top().next_op_idx = op_idx;
               if(debug8count < 140 && optimal_tree){
                  std::println("nextop/opidx increase unregistered. opidx {} is scheduled {} schedulable {}", op_idx -1, working_copy[op_idx-1].is_scheduled,!working_copy.is_schedulable(op_idx-1)); 
               }
               if(op_idx >= sequence.length()){
                  revert_op_idx = true;
                  skip_changes = true;
               }else{
                  continue;
               }
            }

            if(thread_idx >= usable_threads){
               if(debug8count < 140){
                  ////std::println("threadidx {} exceeded usable threads {}", thread_idx, usable_threads);
               }
               revert_thread_idx = true;
               skip_changes = true;
            }

            if(debug8count < 140 && optimal_tree){
               debug8count++;
               std::println("Depth: {}, Op idx: {}, Thread idx: {}, Best makespan: {}, Current makespan: {}", depth, op_idx, thread_idx, best_makespan, makespan);
            }

            if(!skip_changes){
               //old_thread_load = thread_loads[thread_idx];
               working_copy[op_idx].is_scheduled = true;
               const std::size_t start_time = std::max(thread_loads[thread_idx], working_copy.earliest_start(op_idx));
               ////std::println("threadloads: {},threadloads(i): {}, earlieststart: {}, starttime: {}",thread_loads, thread_loads[thread_idx],working_copy.earliest_start(op_idx),start_time);
               working_copy[op_idx].start_time = start_time;
               idling_time += (start_time - thread_loads[thread_idx]);
               thread_loads[thread_idx] = start_time + sequence[op_idx].fma;
               makespan = std::max(makespan, thread_loads[thread_idx]); 
               if(debug8count < 140 && optimal_tree){
                  std::println("{}",working_copy[op_idx]);
               }
            }
            
            if (depth >= sequence.length() - 1) {
                  if (makespan < best_makespan) {
                     //enable for debugging
                     //std::println("Finished a schedule with makespan {} with current best {}", makespan, best_makespan);
                     best_makespan = makespan;
                     for (size_t i = 0; i < sequence.length(); ++i) {
                        sequence[i].thread = working_copy[i].thread;
                        sequence[i].start_time = working_copy[i].start_time;
                        sequence[i].is_scheduled = true;
                     }
                  }
                  revert_thread_idx = true;
            } 
            
            if(!skip_changes && depth < sequence.length() - 1){
                  //If lowed bound still good, go deeper, else revert
                  const std::size_t lb = std::max(((idling_time + sequential_makespan) / usable_threads),working_copy.critical_path());
                  ////std::println("At depth 8, lb {} makespan {}", lb, makespan);
                  if (std::max(lb, makespan) < best_makespan) {
                     working_copy[op_idx].thread = thread_idx;
                     Layer current_layer;
                     current_layer.op_idx = op_idx;
                     current_layer.next_op_idx = 0;
                     current_layer.thread_idx = thread_idx;
                     current_layer.depth = depth++;
                     current_layer.start_time_op = working_copy[op_idx].start_time;
                     current_layer.makespan = makespan;
                     current_layer.idletime = idling_time;
                     current_layer.thread_loads_full = thread_loads;   
                     progress_stack.push(current_layer);
                     op_idx=0;
                     thread_idx=0; 
                  }else{
                     if(debug8count < 140 && optimal_tree){
                        std::println("work/critpath/makespan too high {} / {} / {}, revert thread",((idling_time + sequential_makespan) / usable_threads), working_copy.critical_path(), makespan);
                        if(working_copy.critical_path() > working_copy.sequential_makespan()){
                           std::println("CRIT PATH MISCALCULATION ERROR");
                        }
                     }  
                     revert_thread_idx = true;
                  }
            }

            if(debug8count < 140 && optimal_tree){
               for (size_t i = 0; i < sequence.length(); i++){
                  std::println("op {} start time {} and scheduled {} on t: {}", i, working_copy[i].start_time, working_copy[i].is_scheduled,working_copy[i].thread);
               }

               std::println("threadloads {}", thread_loads);
            }
            

            if(revert_thread_idx){
               if(debug8count < 140 && optimal_tree){
                  ////std::println("revert thread with index {}", thread_idx);
               }
               revert_thread_idx = false;
               Layer previous_state = progress_stack.top();
               if(op_idx < sequence.length()){
                  working_copy[op_idx].start_time = 0;
                  working_copy[op_idx].is_scheduled = false;
               }else{
                  std::println("revert threadidx {} EXCEEDED length {}", op_idx, sequence.length());
               }
               makespan = previous_state.makespan;
               idling_time = previous_state.idletime;
               thread_idx = thread_idx + 1;
               if(thread_idx >= usable_threads){
                  revert_op_idx = true;
               }
               thread_loads = previous_state.thread_loads_full; 
               if(debug8count < 140  && optimal_tree){
                  std::println("REVERT THREAD; threadidx {} nothing else as revertothers {} de {}", thread_idx,revert_op_idx, revert_depth);
               }
            }

            if(revert_op_idx ){
               revert_op_idx = false;
               Layer& previous_state = progress_stack.top();
               if(op_idx < sequence.length()){
                  working_copy[op_idx].start_time = 0;
                  working_copy[op_idx].is_scheduled = false;  
                  previous_state.next_op_idx = previous_state.next_op_idx + 1;
               }else{
                  if(debug8count < 140 && optimal_tree){
                     std::println("revert opidx {} EXCEEDED length {}", op_idx, sequence.length());
                  }
               }
               op_idx = previous_state.next_op_idx;
               thread_idx = 0;
               makespan = previous_state.makespan;
               idling_time = previous_state.idletime;
               if(!progress_stack.empty()){
                  thread_loads = progress_stack.top().thread_loads_full;
               }
               if(op_idx >= sequence.length()){
                  revert_depth = true;
               }
               if(debug8count < 140 && optimal_tree){
                  std::println("REVERT OP; opidx {} in state {}, revert depth set {}", op_idx,previous_state.op_idx, revert_depth);
               }
            }

            if(revert_depth){
               revert_depth = false;
               if(depth == 0){           
                  return true;
               }
               depth--;
               working_copy[progress_stack.top().op_idx].is_scheduled = false;  
               working_copy[progress_stack.top().op_idx].start_time = 0; 
               progress_stack.pop();
               Layer previous_state = progress_stack.top();
               op_idx = previous_state.op_idx;
               thread_idx = previous_state.thread_idx + 1;
               makespan = previous_state.makespan;
               idling_time = previous_state.idletime;
               thread_loads = previous_state.thread_loads_full;
       
               if(debug8count < 140 && optimal_tree){
                  std::println("REVERT DEPTH; depth:{} opidx: {}",depth, op_idx);
                  std::println("mkspn {}, idltme {}, starttime {}, threadloads {}", makespan, idling_time, working_copy[op_idx].start_time,thread_loads);
               }
            }

         }
         return false;
      };


      if(addInitAcc){
         return best_makespan;
      } else {
         addInitAcc = nonrecursive_schedule_op();  
         return best_makespan;
      }
   }
};

}  // namespace jcdp::scheduler

#endif  // JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_