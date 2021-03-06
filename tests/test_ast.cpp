/*
 * @Description: test ast parser.
 * @Author: hunterzju
 * @Date: 2021-12-23 14:43:43
 * @LastEditors: `${env:USERNAME}`
 * @LastEditTime: 2022-01-11 14:39:39
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

module tb_top ();
    reg clk;
    reg reset;
    reg d;
    wire q;

    dff_sync_reset dff_sync_reset_0(.data(d),
                                    .clk(clk),
                                    .reset(reset),
                                    .q(q));
    
    always #10  clk <= ~clk;

    initial begin
        reset <= 0;
        d <= 0;

        #10 reset <= 1;
        #5  d <= 1;
        #8  d <= 0;
        #2  d <= 1;
        #10 d <= 0;
    end

endmodule
    )");
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    SvSyntaxVisitor vis;
    tree->root().visit(vis);
    DialectOpVisitor dial_vis;
    compilation.getRoot().visit(dial_vis);
}
