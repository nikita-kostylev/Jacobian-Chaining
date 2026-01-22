// include/jcdp/scheduler/branch_and_bound_gpu.hpp
#ifndef JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_
#define JCDP_SCHEDULER_BRANCH_AND_BOUND_GPU_HPP_

#include <cstddef>

#include "jcdp/scheduler/scheduler.hpp"
#include "jcdp/sequence.hpp"

namespace jcdp::scheduler {

class BranchAndBoundSchedulerGPU : public Scheduler {
 public:
  auto schedule_impl(
      Sequence& sequence,
      std::size_t usable_threads,
      std::size_t upper_bound) -> std::size_t override final;
};

} // namespace jcdp::scheduler

#endif


