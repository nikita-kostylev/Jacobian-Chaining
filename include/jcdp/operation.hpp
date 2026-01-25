/******************************************************************************
 * @file jcdp/operation.hpp
 *
 * @brief This file is part of the JCDP package. It provides a operations that
 *        can be performed on a Jacobian chain, e.g. eliminations.
 ******************************************************************************/

#ifndef JCDP_OPERATION_HPP_
#define JCDP_OPERATION_HPP_

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INCLUDES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#include <array>
#include <cassert>
#include <compare>  // IWYU pragma: export
#include <cstddef>
#include <cstdint>
#include <format>
#include <string_view>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>> HEADER CONTENTS <<<<<<<<<<<<<<<<<<<<<<<<<<<< //

namespace jcdp {

using std::literals::string_view_literals::operator""sv;

enum class Action : std::uint8_t {
   NONE = 0,
   MULTIPLICATION,
   ACCUMULATION,
   ELIMINATION
};

enum class Mode : std::uint8_t { NONE = 0, TANGENT, ADJOINT };

struct Operation {
   Action action {Action::NONE};
   Mode mode {Mode::NONE};
   std::size_t j {0};
   std::size_t k {0};
   std::size_t i {0};
   std::size_t fma {0};
   std::size_t thread {0};
   std::size_t start_time {0};
   bool is_scheduled {false};
};

inline auto operator>(const Operation& lhs, const Operation& rhs) {
   assert(lhs.action != Action::NONE);
   assert(rhs.action != Action::NONE);
   if (rhs.action != Action::ACCUMULATION) {
      if ((rhs.i == lhs.i && rhs.k == lhs.j) ||
          (rhs.j == lhs.j && rhs.k + 1 == lhs.i)) {
         return true;
      }
   }
   return false;
}

inline auto operator<(const Operation& lhs, const Operation& rhs) {
   assert(lhs.action != Action::NONE);
   assert(rhs.action != Action::NONE);
   if (lhs.action != Action::ACCUMULATION) {
      if ((lhs.i == rhs.i && lhs.k == rhs.j) ||
          (lhs.j == rhs.j && lhs.k + 1 == rhs.i)) {
         return true;
      }
   }
   return false;
}

inline auto operator==(const Operation& lhs, const Operation& rhs) {
   assert(lhs.action != Action::NONE);
   assert(rhs.action != Action::NONE);
   if (lhs.i == rhs.i && lhs.j == rhs.j) {
      return true;
   }
   return false;
}
}  // end namespace jcdp

template<>
struct std::formatter<jcdp::Action> : public std::formatter<std::string_view> {
   template<class FmtContext>
   auto format(const jcdp::Action& action, FmtContext& ctx) const
        -> FmtContext::iterator {
      return std::formatter<std::string_view>::format(
           ACTION_STRINGS[static_cast<std::size_t>(action)], ctx);
   }

   static constexpr std::array ACTION_STRINGS {
        "   "sv, "MUL"sv, "ACC"sv, "ELI"sv};
};

template<>
struct std::formatter<jcdp::Mode> : public std::formatter<std::string_view> {
   template<class FmtContext>
   auto format(const jcdp::Mode& mode, FmtContext& ctx) const
        -> FmtContext::iterator {
      return std::formatter<std::string_view>::format(
           MODE_STRINGS[static_cast<std::size_t>(mode)], ctx);
   }

   static constexpr std::array MODE_STRINGS {"   "sv, "TAN"sv, "ADJ"sv};
};

template<>
struct std::formatter<jcdp::Operation> {
   template<class ParseContext>
   constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
      return ctx.begin();
   }

   template<class FmtContext>
   auto format(const jcdp::Operation& op, FmtContext& ctx) const
        -> FmtContext::iterator {
      // assert(op.action != jcdp::Action::NONE);

      if (op.action == jcdp::Action::ACCUMULATION) {
         assert(op.mode != jcdp::Mode::NONE);

         if (op.mode == jcdp::Mode::TANGENT) {
            return std::format_to(
                 ctx.out(), "{} {} ({:2} {:2}   ) [{}: {} - {}] {}", op.action,
                 op.mode, op.i, op.j + 1, op.thread, op.start_time,
                 op.start_time + op.fma, op.fma);
         } else {
            return std::format_to(
                 ctx.out(), "{} {} (   {:2} {:2}) [{}: {} - {}] {}", op.action,
                 op.mode, op.i, op.j + 1, op.thread, op.start_time,
                 op.start_time + op.fma, op.fma);
         }
      }

      return std::format_to(
           ctx.out(), "{} {} ({:2} {:2} {:2}) [{}: {} - {}] {}", op.action,
           op.mode, op.i, op.k + 1, op.j + 1, op.thread, op.start_time,
           op.start_time + op.fma, op.fma);
   }
};

#endif  // JCDP_OPERATION_HPP_
