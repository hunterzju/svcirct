// Copyright 2021 hunter
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _SVCIRCT_AST_H
#define _SVCIRCT_AST_H

#include <unordered_set>

#include "slang/compilation/Compilation.h"
#include "slang/symbols/ASTVisitor.h"
#include "slang/syntax/SyntaxVisitor.h"

namespace svcirct
{
class SvSyntaxVisitor : public slang::SyntaxVisitor<SvSyntaxVisitor> {
public:
    // LRM: IEEE 1800-2017 
    void handle(const slang::ModuleDeclarationSyntax &mod_decl);
    void handle(const slang::ModuleHeaderSyntax &mod_header);
    
    void handle(const slang::ProceduralBlockSyntax &proc_block);
    void handle(const slang::StatementSyntax &statement); 
    void handle(const slang::TimingControlStatementSyntax &time_ctl);
    void handle(const slang::ConditionalStatementSyntax &cond_stat);
};

class DialectOpVisitor : public slang::ASTVisitor<DialectOpVisitor, true, true> {
public:
    DialectOpVisitor() = default;

    void handle(const slang::InstanceSymbol &inst_symbol);
    void handle(const slang::StatementSyntax &stat_symbol);
};

/// visit slang AST nodes
class ModuleDefinitionVisitor : public slang::ASTVisitor<ModuleDefinitionVisitor, false, false> {
public:
    ModuleDefinitionVisitor() = default;

    // only visit modules
    [[maybe_unused]] void handle(const slang::InstanceSymbol &symbol);

    std::unordered_map<std::string_view, const slang::InstanceSymbol *> modules;
};

/// compute complexity for each module definition. small module will be inlined into its parent
/// module
class ModuleComplexityVisitor : public slang::ASTVisitor<ModuleComplexityVisitor, true, true> {
    // see
    // https://clang.llvm.org/extra/clang-tidy/checks/readability-function-cognitive-complexity.html
public:
    [[maybe_unused]] void handle(const slang::AssignmentExpression &stmt);
    [[maybe_unused]] void handle(const slang::ConditionalStatement &stmt);
    [[maybe_unused]] void handle(const slang::CaseStatement &stmt);
    [[maybe_unused]] void handle(const slang::ForLoopStatement &stmt);

    uint64_t complexity = 0;

private:
    uint64_t current_level_ = 1;
};

class VariableExtractor : public slang::ASTVisitor<VariableExtractor, true, true> {
public:
    VariableExtractor() = default;

    [[maybe_unused]] void handle(const slang::NamedValueExpression &var) { vars.emplace(&var); }

    std::unordered_set<const slang::NamedValueExpression *> vars;
};

class DependencyAnalysisVisitor : public slang::ASTVisitor<DependencyAnalysisVisitor, true, true> {
public:
    struct Node {
        explicit Node(const slang::Symbol &symbol) : symbol(symbol) {}
        // double linked graph
        std::unordered_set<const Node *> edges_to;
        std::unordered_set<const Node *> edges_from;

        const slang::Symbol &symbol;
    };

    struct Graph {
    public:
        std::vector<std::unique_ptr<Node>> nodes;
        std::unordered_map<std::string, Node *> node_mapping;

        Node *get_node(const slang::NamedValueExpression *name);
        Node *get_node(const std::string &name) const;
        Node *get_node(const slang::Symbol &symbol);

    private:
        uint64_t procedural_blk_count_ = 0;
        std::unordered_map<const slang::Symbol *, std::string> new_names_;
    };

    DependencyAnalysisVisitor() : DependencyAnalysisVisitor(nullptr) {}
    explicit DependencyAnalysisVisitor(const slang::Symbol *target);
    DependencyAnalysisVisitor(const slang::Symbol *target, Graph *graph)
        : graph(graph), target_(target) {}

    [[maybe_unused]] void handle(const slang::ContinuousAssignSymbol &stmt);
    [[maybe_unused]] void handle(const slang::ProceduralBlockSymbol &stmt);
    [[maybe_unused]] void handle(const slang::NetSymbol &sym);
    [[maybe_unused]] void handle(const slang::InstanceSymbol &symbol);

    Graph *graph;
    std::vector<const slang::Symbol *> general_always_stmts;

    std::string error;

private:
    std::unique_ptr<Graph> graph_;
    const slang::Symbol *target_;
};

class ProcedureBlockVisitor : public slang::ASTVisitor<ProcedureBlockVisitor, false, false> {
public:
    ProcedureBlockVisitor(const slang::InstanceSymbol *target, slang::ProceduralBlockKind kind)
        : target_(target), kind_(kind) {}

    [[maybe_unused]] void handle(const slang::ProceduralBlockSymbol &stmt);
    [[maybe_unused]] void handle(const slang::InstanceSymbol &symbol);

    std::vector<const slang::ProceduralBlockSymbol *> stmts;

private:
    const slang::InstanceSymbol *target_;
    slang::ProceduralBlockKind kind_;
};
} // namespace svcirct


#endif