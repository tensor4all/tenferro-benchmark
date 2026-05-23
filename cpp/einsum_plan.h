#pragma once

#include <string>
#include <vector>

namespace einsum_plan {

struct ParsedFormat {
    std::vector<std::string> inputs;
    std::string output;
};

ParsedFormat parse_format(const std::string& format);

std::string binary_equation(
    const std::vector<std::string>& subscripts,
    const std::string& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
);

std::vector<std::string> contract_subscripts(
    const std::vector<std::string>& subscripts,
    const std::string& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
);

}  // namespace einsum_plan
