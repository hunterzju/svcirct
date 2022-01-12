/*
 * @Description: file content
 * @Author: hunterzju
 * @Date: 2022-01-12 18:47:14
 * @LastEditors: `${env:USERNAME}`
 * @LastEditTime: 2022-01-12 22:27:58
 * @FilePath: /svcirct/tests/test_mlir.cpp
 */
#include "../src/ast.h"
#include "gtest/gtest.h"
#include "slang/syntax/SyntaxTree.h"

using namespace svcirct;
using namespace slang;

TEST(ast, instance) {  // NOLINT
    auto tree = SyntaxTree::fromText(R"(
module module();
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    DialectOpVisitor vis;
    compilation.getRoot().visit(vis);
}