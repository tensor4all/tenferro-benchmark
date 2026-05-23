#include "einsum_plan.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace einsum_plan {

namespace {

std::vector<std::string> split_inputs(const std::string& inputs) {
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= inputs.size()) {
        const auto comma = inputs.find(',', start);
        if (comma == std::string::npos) {
            result.push_back(inputs.substr(start));
            break;
        }
        result.push_back(inputs.substr(start, comma - start));
        start = comma + 1;
    }
    return result;
}

bool contains_char(const std::string& text, char c) {
    return text.find(c) != std::string::npos;
}

void append_unique(std::string& text, char c) {
    if (!contains_char(text, c)) {
        text.push_back(c);
    }
}

std::string intermediate_output(
    const std::vector<std::string>& subscripts,
    const std::string& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    if (lhs_index >= subscripts.size() || rhs_index >= subscripts.size()) {
        throw std::out_of_range("contraction index out of range");
    }
    if (lhs_index == rhs_index) {
        throw std::invalid_argument("cannot contract an operand with itself");
    }

    std::unordered_map<char, std::size_t> outside_counts;
    for (std::size_t i = 0; i < subscripts.size(); ++i) {
        if (i == lhs_index || i == rhs_index) {
            continue;
        }
        for (char c : subscripts[i]) {
            outside_counts[c] += 1;
        }
    }

    std::string labels_in_order;
    for (char c : subscripts[lhs_index]) {
        append_unique(labels_in_order, c);
    }
    for (char c : subscripts[rhs_index]) {
        append_unique(labels_in_order, c);
    }

    std::string output;
    for (char c : final_output) {
        if (contains_char(labels_in_order, c)) {
            append_unique(output, c);
        }
    }
    for (char c : labels_in_order) {
        if (outside_counts[c] > 0) {
            append_unique(output, c);
        }
    }
    return output;
}

}  // namespace

ParsedFormat parse_format(const std::string& format) {
    const auto arrow = format.find("->");
    if (arrow == std::string::npos) {
        throw std::invalid_argument("einsum format must contain ->");
    }
    ParsedFormat parsed;
    parsed.inputs = split_inputs(format.substr(0, arrow));
    parsed.output = format.substr(arrow + 2);
    return parsed;
}

std::string binary_equation(
    const std::vector<std::string>& subscripts,
    const std::string& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    const auto output = intermediate_output(subscripts, final_output, lhs_index, rhs_index);
    return subscripts[lhs_index] + "," + subscripts[rhs_index] + "->" + output;
}

std::vector<std::string> contract_subscripts(
    const std::vector<std::string>& subscripts,
    const std::string& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    const auto output = intermediate_output(subscripts, final_output, lhs_index, rhs_index);
    auto next = subscripts;
    const auto first_remove = std::max(lhs_index, rhs_index);
    const auto second_remove = std::min(lhs_index, rhs_index);
    next.erase(next.begin() + static_cast<std::ptrdiff_t>(first_remove));
    next.erase(next.begin() + static_cast<std::ptrdiff_t>(second_remove));
    next.push_back(output);
    return next;
}

}  // namespace einsum_plan
