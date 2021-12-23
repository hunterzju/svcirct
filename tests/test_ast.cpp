/*
 * @Description: test ast parser.
 * @Author: hunterzju
 * @Date: 2021-12-23 14:43:43
 * @LastEditors: `${env:USERNAME}`
 * @LastEditTime: 2021-12-23 17:25:01
 * @FilePath: /svcirct/tests/test_ast.cpp
 */
#include "../src/ast.h"
#include "gtest/gtest.h"
#include "slang/syntax/SyntaxTree.h"

using namespace svcirct;
using namespace slang;

TEST(ast, instance) {  // NOLINT
    auto tree = SyntaxTree::fromText(R"(
module m1;
endmodule
module m2;
endmodule
module m3;
m1 m1_();
endmodule
module m;
    m1 m1_();
    m2 m2_();
    m3 m3_();
endmodule
)");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    ModuleDefinitionVisitor vis;
    compilation.getRoot().visit(vis);
    EXPECT_EQ(vis.modules.size(), 4);
}


