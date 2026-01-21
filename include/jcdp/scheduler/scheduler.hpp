/******************************************************************************
 * @file jcdp/scheduler/scheduler.hpp
 *
 * @brief This file is part of the JCDP package. It provides a base class
 *        for a scheduler that assigns threads and start times to operations
 *        in a given elimination sequence.
 ******************************************************************************/

#ifndef JCDP_SCHEDULER_SCHEDULER_HPP_
#define JCDP_SCHEDULER_SCHEDULER_HPP_

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <cstddef>
#include <limits>
#include <print>

#include "jcdp/deviceSequence.hpp"
#include "jcdp/sequence.hpp"
#include "jcdp/util/timer.hpp"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp::scheduler {

class Scheduler : public util::Timer {
 public:
   Scheduler() = default;
   virtual ~Scheduler() = default;

   inline auto schedule(
        DeviceSequence& sequence, const std::size_t threads,
        const std::size_t upper_bound = std::numeric_limits<std::size_t>::max())
        -> std::size_t {

      start_timer();

      // We can never use more threads than we have accumulations
      std::size_t usable_threads = count_accumulations(sequence);
      if (threads > 0 && threads < usable_threads) {
         usable_threads = threads;
      }

      return schedule_impl(sequence, usable_threads, upper_bound);
   }

   virtual auto schedule_impl(DeviceSequence&, const std::size_t, const std::size_t)
        -> std::size_t = 0;
};

}  // namespace jcdp::scheduler

#endif  // JCDP_SCHEDULER_SCHEDULER_HPP_
