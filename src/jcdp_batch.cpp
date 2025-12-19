/******************************************************************************
 * @file jcdp_batch.cpp
 *
 * @brief This file is part of the JCDP package. It provides an application that
 *        generated multiple Jacobian chains and runs all available solvers on
 *        them. The makespan of the calculated sequences are stored in CSV
 *        files. The generator and solver properties can be provided via a
 *        config files that is expected as the first command line argument.
 ******************************************************************************/

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "jcdp/generator.hpp"
#include "jcdp/jacobian_chain.hpp"
#include "jcdp/optimizer/branch_and_bound.hpp"
#include "jcdp/optimizer/dynamic_programming.hpp"
#include "jcdp/scheduler/branch_and_bound.hpp"
#include "jcdp/scheduler/branch_and_bound_gpu.hpp"
#include "jcdp/scheduler/priority_list.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> APPLICATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

int main(int argc, char* argv[]) {
   jcdp::JacobianChainGenerator jcgen;
   jcdp::optimizer::DynamicProgrammingOptimizer dp_solver;
   jcdp::optimizer::BranchAndBoundOptimizer bnb_solver;

   std::shared_ptr<jcdp::scheduler::BranchAndBoundScheduler> bnb_scheduler =
        std::make_shared<jcdp::scheduler::BranchAndBoundScheduler>();
      std::shared_ptr<jcdp::scheduler::BranchAndBoundSchedulerGPU>
         bnb_scheduler_gpu =
            std::make_shared<jcdp::scheduler::BranchAndBoundSchedulerGPU>();
   std::shared_ptr<jcdp::scheduler::PriorityListScheduler> list_scheduler =
        std::make_shared<jcdp::scheduler::PriorityListScheduler>();

   if (argc < 2) {
      jcgen.print_help(std::cout);
      dp_solver.print_help(std::cout);
      return -1;
   }

   const std::filesystem::path config_filename(argv[1]);
   try {
      dp_solver.parse_config(config_filename, true);
      bnb_solver.parse_config(config_filename, true);
      jcgen.parse_config(config_filename, true);
      jcgen.init_rng();
   } catch (const std::runtime_error& bcfe) {
      std::println(std::cerr, "{}", bcfe.what());
      return -1;
   }

   std::string output_file_name = "results";
   if (argc > 2) {
      output_file_name = argv[2];
   }

   jcdp::JacobianChain chain;
   while (!jcgen.empty()) {
      const std::size_t len = jcgen.current_length();
      std::filesystem::path output_file =
           (output_file_name + std::to_string(len) + ".csv");

      std::ofstream out(output_file);
      if (!out) {
         std::println(std::cerr, "Failed to open {}", output_file.string());
         return -1;
      }

      for (std::size_t t = 1; t <= len; ++t) {
         std::print(out, "BnB_BnB/{}/finished,", t);
         std::print(out, "BnB_BnB/{},", t);
         std::print(out, "BnB_BnB_GPU/{}/finished,", t);
         std::print(out, "BnB_BnB_GPU/{},", t);
         std::print(out, "BnB_List/{},", t);
         std::print(out, "DP/{},", t);
         std::print(out, "DP_BnB/{}{}", t, (t < len) ? "," : "\n");
      }

      while (jcgen.next(chain)) {
         chain.init_subchains();

         // Solve via dynamic programming
         dp_solver.init(chain);
         dp_solver.m_usable_threads = len;
         dp_solver.solve();

         for (std::size_t t = 1; t <= len; ++t) {
            jcdp::Sequence dp_seq = dp_solver.get_sequence(t);
            const std::size_t dp_makespan = dp_seq.makespan();

            // Schedule dynamic programming sequence via branch & bound
            bnb_scheduler->schedule(dp_seq, t, dp_makespan);

            // Solve via branch & bound + List scheduling
            bnb_solver.init(chain, list_scheduler);
            bnb_solver.set_upper_bound(dp_seq.makespan());
            bnb_solver.m_usable_threads = t;
            jcdp::Sequence bnb_seq_list = bnb_solver.solve();

            // Solve via branch & bound + branch & bound scheduling
            bnb_solver.init(chain, bnb_scheduler);
            bnb_solver.set_upper_bound(bnb_seq_list.makespan());
            bnb_solver.m_usable_threads = t;
            jcdp::Sequence bnb_seq = bnb_solver.solve();
            const bool finished_bnb = bnb_solver.finished_in_time();

            // Solve via branch & bound + branch & bound GPU scheduling
            bnb_solver.init(chain, bnb_scheduler_gpu);
            bnb_solver.set_upper_bound(bnb_seq_list.makespan());
            bnb_solver.m_usable_threads = t;
            jcdp::Sequence bnb_seq_gpu = bnb_solver.solve();
            const bool finished_bnb_gpu = bnb_solver.finished_in_time();

            std::print(out, "{},", finished_bnb);
            std::print(out, "{},", bnb_seq.makespan());
            std::print(out, "{},", finished_bnb_gpu);
            std::print(out, "{},", bnb_seq_gpu.makespan());
            std::print(out, "{},", bnb_seq_list.makespan());
            std::print(out, "{},", dp_makespan);
            std::print(out, "{}{}", dp_seq.makespan(),
                       (t < len) ? "," : "\n");
         }

         out.flush();
      }

      out.close();
   }

   return 0;
}
