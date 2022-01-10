/*
 * @Description: test ast parser.
 * @Author: hunterzju
 * @Date: 2021-12-23 14:43:43
 * @LastEditors: `${env:USERNAME}`
 * @LastEditTime: 2022-01-09 14:58:11
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

TEST(ast, module) {
    auto tree = SyntaxTree::fromText(R"(
module dff_sync_reset (
input  wire data  ,
input  wire clk   ,
input  wire reset ,
output reg  q);
always_ff @ ( posedge clk)
if (~reset) begin
  q <= 1'b0;
end  else begin
  q <= data;
end
endmodule
    )");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    SvSyntaxVisitor vis;
    tree->root().visit(vis);
    GlobalVisitor g_vis;
    compilation.getRoot().visit(g_vis);
}
