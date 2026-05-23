#include "einsum_plan.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect_eq(const std::string& got, const std::string& want, const char* label) {
    if (got != want) {
        std::cerr << label << ": got `" << got << "`, want `" << want << "`\n";
        std::exit(1);
    }
}

void expect_vec_eq(
    const std::vector<std::string>& got,
    const std::vector<std::string>& want,
    const char* label
) {
    if (got != want) {
        std::cerr << label << ": got [";
        for (const auto& s : got) {
            std::cerr << s << " ";
        }
        std::cerr << "], want [";
        for (const auto& s : want) {
            std::cerr << s << " ";
        }
        std::cerr << "]\n";
        std::exit(1);
    }
}

void test_parse_format() {
    auto parsed = einsum_plan::parse_format("ij,jk->ik");
    expect_vec_eq(parsed.inputs, {"ij", "jk"}, "parse inputs");
    expect_eq(parsed.output, "ik", "parse output");
}

void test_binary_equation_for_matmul() {
    auto parsed = einsum_plan::parse_format("ij,jk->ik");
    auto equation = einsum_plan::binary_equation(parsed.inputs, parsed.output, 0, 1);
    expect_eq(equation, "ij,jk->ik", "matmul equation");
}

void test_binary_equation_keeps_external_labels() {
    auto parsed = einsum_plan::parse_format("ab,bc,cd->ad");
    auto first = einsum_plan::binary_equation(parsed.inputs, parsed.output, 0, 1);
    expect_eq(first, "ab,bc->ac", "first chain equation");

    auto after = einsum_plan::contract_subscripts(parsed.inputs, parsed.output, 0, 1);
    expect_vec_eq(after, {"cd", "ac"}, "updated subscripts");

    auto second = einsum_plan::binary_equation(after, parsed.output, 0, 1);
    expect_eq(second, "cd,ac->ad", "second chain equation");
}

}  // namespace

int main() {
    test_parse_format();
    test_binary_equation_for_matmul();
    test_binary_equation_keeps_external_labels();
    return 0;
}
