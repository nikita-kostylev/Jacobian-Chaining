/******************************************************************************
 * @file jcdp/optimizer/dynamic_programming.hpp
 *
 * @brief This file is part of the JCDP package. It provides an optimizer that
 *        uses a dynamic programming algorithm to find the best possible
 *        brackating (elimination sequence) for a given Jacobian chain.
 ******************************************************************************/

#ifndef JCDP_OPTIMIZER_BRANCH_AND_BOUND_HPP_
#define JCDP_OPTIMIZER_BRANCH_AND_BOUND_HPP_

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <utility>
#include <vector>

#include "jcdp/deviceSequence.hpp"
#include "jcdp/jacobian.hpp"
#include "jcdp/jacobian_chain.hpp"
#include "jcdp/operation.hpp"
#include "jcdp/optimizer/optimizer.hpp"
#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"
#include "jcdp/util/timer.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::optimizer {

class BranchAndBoundOptimizer : public Optimizer, public util::Timer {
   using OpPair = std::array<std::optional<Operation>, 2>;

 public:
   BranchAndBoundOptimizer() : Optimizer() {
      register_property(
           m_time_to_solve, "time_to_solve",
           "Maximal runtime for the branch & bound solver in seconds.");
   }

   virtual ~BranchAndBoundOptimizer() = default;

   auto init(
        const JacobianChain& chain,
        const std::shared_ptr<scheduler::Scheduler>& sched) -> void {
      Optimizer::init(chain);

      m_scheduler = sched;
      m_optimal_sequence = Sequence::make_max();
      m_makespan = m_optimal_sequence.makespan();
      m_upper_bound = m_makespan;
      m_timer_expired = false;

      m_leafs = 0;
      m_updated_makespan = 0;
      m_pruned_branches.clear();
      m_pruned_branches.resize(m_chain.longest_possible_sequence() + 1);
   }

   virtual auto solve() -> Sequence override final {

      set_timer(m_time_to_solve);
      start_timer();
      std::size_t accs = m_matrix_free ? 0 : (m_length - 1);

#pragma omp parallel default(shared)
#pragma omp single
      while (++accs <= m_length) {
         Sequence sequence {};
         std::vector<OpPair> eliminations {};
         JacobianChain chain = m_chain;
         add_accumulation(sequence, chain, accs, eliminations);
      }
      return m_optimal_sequence;
   }

   inline auto set_upper_bound(const std::size_t upper_bound) {
      m_upper_bound = upper_bound;
   }

   inline auto print_stats() -> void {
      std::println("Leafs visited (= sequences scheduled): {}", m_leafs);
      std::println("Updated makespan: {}", m_updated_makespan);
      std::println(
           "Pruned branches: {}",
           std::reduce(m_pruned_branches.cbegin(), m_pruned_branches.cend()));
      std::println("Pruned branches per sequence length:");
      std::print("[ ");
      for (const std::size_t pruned : m_pruned_branches) {
         std::print("{} ", pruned);
      }
      std::println("]");
   }

 private:
   DeviceSequence dev_m_optimal_sequence {device_make_max()};
   std::size_t dev_m_makespan {makespan(dev_m_optimal_sequence)};

   Sequence m_optimal_sequence {Sequence::make_max()};
   std::size_t m_makespan {m_optimal_sequence.makespan()};
   std::size_t m_upper_bound {m_makespan};
   std::size_t m_leafs {0};
   std::vector<std::size_t> m_pruned_branches {};
   std::size_t m_updated_makespan {0};
   std::shared_ptr<scheduler::Scheduler> m_scheduler;

   using Optimizer::init;

   inline auto add_accumulation(
        Sequence& sequence, JacobianChain& chain, const std::size_t accs,
        std::vector<OpPair>& eliminations, std::size_t j = 0) -> void {
      if (accs > 0) {
         for (; j < m_chain.length(); ++j) {
            const Operation op = cheapest_accumulation(j);
            if (!chain.apply(op)) {
               continue;
            }

            push_possible_eliminations(chain, eliminations, op.j, op.i);
            sequence.push_back(std::move(op));

            add_accumulation(sequence, chain, accs - 1, eliminations, j + 1);

            sequence.pop_back();
            eliminations.pop_back();
            chain.revert(op);
         }
      } else {
         // Copies for spawned task (Necessary on Windows)
         Sequence task_sequence = sequence;
         JacobianChain task_chain = chain;
         std::vector<OpPair> task_eliminations = eliminations;

#pragma omp task default(none) firstprivate(task_sequence)                     \
     firstprivate(task_chain, task_eliminations)
         add_elimination(task_sequence, task_chain, task_eliminations);
      }
   }

   inline jcdp::DeviceSequence to_device(const jcdp::Sequence& seq) {
      jcdp::DeviceSequence d {};
      d.length = seq.length();
      std::println("Actual size of sequence is: %d", d);

      for (std::size_t i = 0; i < d.length; ++i) {
         d.ops[i] = seq[i];
      }

      return d;
   }

   inline auto add_elimination(
        Sequence& sequence, JacobianChain& chain,
        std::vector<OpPair>& eliminations, std::size_t elim_idx = 0) -> void {

      // Return if time's up
      if (!remaining_time()) {
         return;
      }

      // Check if we accumulated the entire jacobian
      if (chain.get_jacobian(chain.length() - 1, 0).is_accumulated) {
         assert(elim_idx == eliminations.size() - 1);
         assert(!eliminations[elim_idx][0].has_value());
         assert(!eliminations[elim_idx][1].has_value());

         // Start new task for the scheduling of the final sequence. If
         // branch & bound is used as the scheduling algorithm, this can take
         // some time.

         // Copies for spawned task (Necessary on Windows)
         Sequence final_sequence = sequence;
         const std::shared_ptr<scheduler::Scheduler> scheduler = m_scheduler;

         DeviceSequence deviceSequence = to_device(sequence);

#pragma omp task default(shared) firstprivate(deviceSequence, scheduler)
         {
            const double time_to_schedule = remaining_time();
            if (time_to_schedule) {
               scheduler->set_timer(time_to_schedule);

               const std::size_t new_makespan = scheduler->schedule(
                    deviceSequence, m_usable_threads, m_makespan);

               m_timer_expired |= !scheduler->finished_in_time();

#pragma omp atomic
               m_leafs++;

#pragma omp critical
               if (m_makespan > new_makespan) {
                  dev_m_optimal_sequence = ;
                  m_optimal_sequence = deviceSequence;
                  m_makespan = new_makespan;
                  m_updated_makespan++;
               }
            }
         }
         return;
      }

      // Check critical path as lower bound
      const std::size_t lower_bound = sequence.critical_path();
      if (lower_bound >= m_makespan || lower_bound > m_upper_bound) {
         std::size_t& prune_counter = m_pruned_branches[sequence.length()];

#pragma omp atomic
         prune_counter++;

         return;
      }

      // Perform all possible elimination from the current elim_idx
      for (; elim_idx < eliminations.size(); ++elim_idx) {
         for (std::size_t pair_idx = 0; pair_idx <= 1; ++pair_idx) {
            if (!eliminations[elim_idx][pair_idx].has_value()) {
               continue;
            }

            const Operation op = eliminations[elim_idx][pair_idx].value();
            if (!chain.apply(op)) {
               continue;
            }

            push_possible_eliminations(chain, eliminations, op.j, op.i);
            sequence.push_back(op);

            add_elimination(sequence, chain, eliminations, elim_idx + 1);

            sequence.pop_back();
            eliminations.pop_back();
            chain.revert(op);
         }
      }
   }

   inline auto cheapest_accumulation(const std::size_t j) -> Operation {
      const Jacobian& jac = m_chain.get_jacobian(j, j);
      Operation op {
           .action = Action::ACCUMULATION,
           .mode = Mode::TANGENT,
           .j = j,
           .k = j,
           .i = j,
           .fma = jac.fma<Mode::TANGENT>()};

      if (m_available_memory == 0 || m_available_memory >= jac.edges_in_dag) {
         const std::size_t adjoint_fma = jac.fma<Mode::ADJOINT>();
         if (adjoint_fma < op.fma) {
            op.mode = Mode::ADJOINT;
            op.fma = adjoint_fma;
         }
      }

      return op;
   }

   inline auto push_possible_eliminations(
        const JacobianChain& chain, std::vector<OpPair>& eliminations,
        const std::size_t op_j, const std::size_t op_i) -> void {
      OpPair ops {};

      // Tangent or multiplication
      if (op_j < chain.length() - 1) {
         const std::size_t k = op_j;
         const std::size_t i = op_i;
         const Jacobian& ki_jac = chain.get_jacobian(k, i);

         // Add multiplication if possible
         std::size_t j;
         for (j = m_chain.length() - 1; j >= k + 1; --j) {
            const Jacobian& jk_jac = chain.get_jacobian(j, k + 1);
            if (!jk_jac.is_accumulated || jk_jac.is_used) {
               continue;
            }

            ops[0] = Operation {
                 .action = Action::MULTIPLICATION,
                 .j = j,
                 .k = k,
                 .i = i,
                 .fma = jk_jac.m * ki_jac.m * ki_jac.n};

            break;
         }

         // Add tangent elimination if multiplication wasn't possible
         if (k + 1 == ++j && m_matrix_free) {
            const Jacobian& jk_jac = chain.get_jacobian(j, k + 1);
            assert(!jk_jac.is_accumulated && !jk_jac.is_used);

            ops[0] = Operation {
                 .action = Action::ELIMINATION,
                 .mode = Mode::TANGENT,
                 .j = j,
                 .k = k,
                 .i = i,
                 .fma = jk_jac.fma<Mode::TANGENT>(ki_jac.n)};
         }
      }

      // Adjoint or multiplication
      if (op_i > 0) {
         const std::size_t k = op_i - 1;
         const std::size_t j = op_j;
         const Jacobian& jk_jac = chain.get_jacobian(j, k + 1);

         // Add multiplication if possible
         std::size_t i;
         for (i = 0; i <= k; ++i) {
            const Jacobian& ki_jac = chain.get_jacobian(k, i);
            if (!ki_jac.is_accumulated || ki_jac.is_used) {
               continue;
            }

            ops[1] = Operation {
                 .action = Action::MULTIPLICATION,
                 .j = j,
                 .k = k,
                 .i = i,
                 .fma = jk_jac.m * ki_jac.m * ki_jac.n};
            break;
         }

         // Add adjoint elimination if multiplication wasn't possible
         if (k == --i && m_matrix_free) {
            const Jacobian& ki_jac = chain.get_jacobian(k, i);
            assert(!ki_jac.is_accumulated && !ki_jac.is_used);

            if (m_available_memory == 0 ||
                m_available_memory >= ki_jac.edges_in_dag) {
               ops[1] = Operation {
                    .action = Action::ELIMINATION,
                    .mode = Mode::ADJOINT,
                    .j = j,
                    .k = k,
                    .i = i,
                    .fma = ki_jac.fma<Mode::ADJOINT>(jk_jac.m)};
            }
         }
      }

      eliminations.push_back(ops);
   }
};

}  // namespace jcdp::optimizer

#endif  // JCDP_OPTIMIZER_BRANCH_AND_BOUND_HPP_
