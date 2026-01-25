/******************************************************************************
 * @file jcdp.cpp
 *
 * @brief This file is part of the JCDP package. It provides an applications
 *        that generated Jacobian chains based on a given config file and runs
 *        dynamic programming, and Branch & Bound optimizers combined with
 *        a list scheduler and a Branch & Bound scheduler.
 ******************************************************************************/

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

#include "jcdp/generator.hpp"
#include "jcdp/jacobian_chain.hpp"
#include "jcdp/operation.hpp"
#include "jcdp/optimizer/bnb_block.hpp"
#include "jcdp/optimizer/branch_and_bound.hpp"
#include "jcdp/optimizer/dynamic_programming.hpp"
#include "jcdp/scheduler/bnb_block.hpp"
#include "jcdp/scheduler/branch_and_bound.hpp"
#include "jcdp/scheduler/branch_and_bound_gpu.hpp"
#include "jcdp/scheduler/priority_list.hpp"
#include "jcdp/sequence.hpp"
#include "jcdp/util/dot_writer.hpp"
#include "omp.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> APPLICATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

int main(int argc, char* argv[]) {
   std::println("Devices: {}", omp_get_num_devices());
   std::println("Threads: {}", omp_get_max_threads());

   jcdp::JacobianChainGenerator jcgen;
   jcdp::optimizer::DynamicProgrammingOptimizer dp_solver;
   jcdp::optimizer::BranchAndBoundOptimizer bnb_solver;
   jcdp::optimizer::BnBBlockOptimizer bnb_block_solver;

   jcdp::scheduler::PriorityListScheduler list_scheduler =
        jcdp::scheduler::PriorityListScheduler();
   jcdp::scheduler::PriorityListScheduler* list_s_p = &list_scheduler;

   jcdp::scheduler::BranchAndBoundScheduler bnb_scheduler =
        jcdp::scheduler::BranchAndBoundScheduler();
   jcdp::scheduler::BranchAndBoundScheduler* bnb_s_p = &bnb_scheduler;

   jcdp::scheduler::BnBBlockScheduler bnb_block_scheduler =
        jcdp::scheduler::BnBBlockScheduler();
   jcdp::scheduler::BnBBlockScheduler* bnb_b_s_p = &bnb_block_scheduler;

   jcdp::scheduler::BranchAndBoundSchedulerGPU bnb_scheduler_gpu =
        jcdp::scheduler::BranchAndBoundSchedulerGPU();
   jcdp::scheduler::BranchAndBoundSchedulerGPU* bnb_s_g_p = &bnb_scheduler_gpu;

   if (argc < 2) {
      jcgen.print_help(std::cout);
      dp_solver.print_help(std::cout);
      return -1;
   }

   const std::filesystem::path config_filename(argv[1]);
   try {
      dp_solver.parse_config(config_filename, true);
      bnb_solver.parse_config(config_filename, true);
      bnb_block_solver.parse_config(config_filename, true);
      jcgen.parse_config(config_filename, true);
      jcgen.init_rng();
   } catch (const std::runtime_error& bcfe) {
      std::println(std::cerr, "{}", bcfe.what());
      return -1;
   }

   std::println("Chain generator properties:");
   jcgen.print_values(std::cout);

   std::println("\ndp_solver properties:");
   dp_solver.print_values(std::cout);

   jcdp::JacobianChain chain;
   jcgen.next(chain);
   chain.init_subchains();

   std::println(
        "\nTangent cost: {}",
        chain.get_jacobian(chain.length() - 1, 0).fma<jcdp::Mode::TANGENT>());
   std::println(
        "Adjoint cost: {}",
        chain.get_jacobian(chain.length() - 1, 0).fma<jcdp::Mode::ADJOINT>());

   // Solve via dynamic programming
   dp_solver.init(chain);
   auto start_dp = std::chrono::high_resolution_clock::now();
   jcdp::Sequence dp_seq = dp_solver.solve();
   auto end_dp = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration_dp = end_dp - start_dp;
   std::println("\nDP solve duration: {} seconds", duration_dp.count());
   std::println("Optimized cost (DP): {}\n", dp_seq.makespan());
   std::println("{}", dp_seq);

   jcdp::util::write_dot(dp_seq, "dynamic_programming");

   if (true) {

      // Schedule dynamic programming sequence via list scheduling
      auto start_list_sched = std::chrono::high_resolution_clock::now();
      list_s_p->schedule(dp_seq, dp_solver.m_usable_threads);
      auto end_list_sched = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_list_sched = end_list_sched -
                                                          start_list_sched;
      std::println(
           "\nScheduling duration: {} seconds", duration_list_sched.count());
      std::println(
           "Optimized cost (DP + List scheduling): {}\n", dp_seq.makespan());
      std::println("{}", dp_seq);

      // Schedule dynamic programming sequence via branch & bound
      auto start_sched = std::chrono::high_resolution_clock::now();
      bnb_s_p->schedule(dp_seq, dp_solver.m_usable_threads);
      auto end_sched = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_sched = end_sched - start_sched;
      std::println("\nScheduling duration: {} seconds", duration_sched.count());
      std::println(
           "Optimized cost (DP + B&B scheduling): {}\n", dp_seq.makespan());
      std::println("{}", dp_seq);

      // Solve via branch & bound + List scheduling
      bnb_solver.init(chain, list_s_p);
      bnb_solver.set_upper_bound(dp_seq.makespan());
      auto start_bnb_list = std::chrono::high_resolution_clock::now();
      jcdp::Sequence bnb_seq_list = bnb_solver.solve();
      auto end_bnb_list = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_bnb_list = end_bnb_list -
                                                        start_bnb_list;
      std::println(
           "\nBnB (List) solve duration: {} seconds",
           duration_bnb_list.count());
      bnb_solver.print_stats();
      std::println(
           "Optimized cost (BnB + List scheduling): {}\n",
           bnb_seq_list.makespan());
      std::println("{}", bnb_seq_list);

      // Solve via branch & bound
      bnb_solver.init(chain, bnb_s_p);
      bnb_solver.set_upper_bound(dp_seq.makespan());
      auto start_bnb = std::chrono::high_resolution_clock::now();
      jcdp::Sequence bnb_seq = bnb_solver.solve();
      auto end_bnb = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_bnb = end_bnb - start_bnb;
      std::println("\nBnB solve duration: {} seconds", duration_bnb.count());
      bnb_solver.print_stats();
      std::println("Optimized cost (BnB): {}\n", bnb_seq.makespan());
      std::println("{}", bnb_seq);

      jcdp::util::write_dot(bnb_seq, "branch_and_bound");
   }

   // Solve via branch & bound (GPU branch & bound scheduler)
   bnb_solver.init(chain, bnb_s_g_p);
   bnb_solver.set_makespan(SIZE_MAX);
   auto start_bnb_gpu = std::chrono::high_resolution_clock::now();
   jcdp::Sequence bnb_seq_gpu = bnb_solver.solve();
   auto end_bnb_gpu = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration_bnb_gpu = end_bnb_gpu - start_bnb_gpu;
   std::println(
        "\nBnB (GPU sched) solve duration: {} seconds",
        duration_bnb_gpu.count());
   bnb_solver.print_stats();
   std::println(
        "Optimized cost (BnB + GPU sched): {}\n", bnb_seq_gpu.makespan());

   std::println("{}", bnb_seq_gpu);

   jcdp::util::write_dot(bnb_seq_gpu, "branch_and_bound_gpu");

   if (true) {
      // Schedule dynamic programming sequence via GPU branch & bound scheduling
      auto start_sched = std::chrono::high_resolution_clock::now();
      bnb_s_g_p->set_timer(1200);
      bnb_s_g_p->start_timer();
      bnb_s_g_p->schedule(dp_seq, dp_solver.m_usable_threads);
      auto end_sched = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_sched = end_sched - start_sched;
      std::println("\nScheduling duration: {} seconds", duration_sched.count());
      std::println(
           "Optimized cost (DP + B&B GPU scheduling ): {}\n",
           dp_seq.makespan());

      std::println("{}", dp_seq);
   }

   // Solve via branch & bound block
   if (true) {  // disabled, as we only test simple solution
      bnb_block_solver.init(chain, bnb_s_g_p);
      // remove FOR MVP
      // bnb_block_solver.set_upper_bound(bnb_seq_list.makespan());
      auto start_bnb_block = std::chrono::high_resolution_clock::now();
      jcdp::Sequence bnb_seq_block = bnb_block_solver.solve();
      auto end_bnb_block = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_bnb_block = end_bnb_block -
                                                         start_bnb_block;
      std::println(
           "\nBnB Block solve duration: {} seconds",
           duration_bnb_block.count());
      bnb_block_solver.print_stats();
      std::println("Optimized cost (BnB): {}\n", bnb_seq_block.makespan());
      std::println("{}", bnb_seq_block);

      jcdp::util::write_dot(bnb_seq_block, "branch_and_bound");
   }

   return 0;
}
