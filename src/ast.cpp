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

#include "ast.h"

#include <unordered_set>
#include <string>

namespace svcirct {

// LRM 23.2
void SvSyntaxVisitor::handle(const slang::ModuleDeclarationSyntax &mod_decl) {
    fmt::print("visit module def: {}\n", mod_decl.toString());
    visit(*mod_decl.header);
    
    for(auto child : mod_decl.members) {
        fmt::print("module child node type: {}\n", slang::toString(child->kind));
        visit(*child);
    }
}

void SvSyntaxVisitor::handle(const slang::ModuleHeaderSyntax &mod_header) {
    fmt::print("visit module header: {}\n", mod_header.toString());
}

// LRM 9.2.2
void SvSyntaxVisitor::handle(const slang::ProceduralBlockSyntax &proc_block) {
    fmt::print("visit proc block: {}\n{}\n", slang::toString(proc_block.kind), proc_block.toString());
    visit(*proc_block.statement);
}

// LRM 9.4 Handle TimingControlStatement
void SvSyntaxVisitor::handle(const slang::TimingControlStatementSyntax &time_ctl) {
    fmt::print("visit time_ctrl_statement: {}\n", time_ctl.toString());
    fmt::print("time ctrl kind: {}\n{}\n", slang::toString(time_ctl.timingControl->kind), time_ctl.timingControl->toString());
    fmt::print("time ctrl sub statement kind: {}\n{}\n", slang::toString(time_ctl.statement->kind), time_ctl.statement->toString());
    visit(*time_ctl.timingControl);
    visit(*time_ctl.statement);
}

// LRM 12.4
void SvSyntaxVisitor::handle(const slang::ConditionalStatementSyntax &cond_stat) {
    fmt::print("visit condition statement: {}\n", cond_stat.toString());
    fmt::print("cond predicate: {}\n{}\n", slang::toString(cond_stat.predicate->kind), cond_stat.predicate->toString());
    fmt::print("cond statement: {}\n{}\n", slang::toString(cond_stat.statement->kind), cond_stat.statement->toString());
    fmt::print("cond elseClause: {}\n{}", slang::toString(cond_stat.elseClause->kind), cond_stat.elseClause->toString());
    visit(*cond_stat.statement);
}

void SvSyntaxVisitor::handle(const slang::StatementSyntax &statement) {
    fmt::print("visit statement: {}\n{}\n", slang::toString(statement.kind), statement.toString());

    switch (statement.kind)
    {
        case slang::SyntaxKind::ConditionalStatement:{
            /* need explicit {} */
            fmt::print("visit cond statement: {}\n", statement.toString());
            auto cond_stat = static_cast<const slang::ConditionalStatementSyntax*>(&statement);
            visit(*cond_stat);
            break;
        }
        default:
            fmt::print("statement kind: {}\n", statement.kind);
            break;
    }
}
    
    void NodeVisitor::handle(const slang::InstanceSymbol &inst_symbol) {
        fmt::print("visit module instance: {}\n", inst_symbol.name);
    }

    void NodeVisitor::handle(const slang::VariableSymbol &var_symbol) {    
        fmt::print("visit module instance: {}\n", var_symbol.name);
    }

    void NodeVisitor::handle(const slang::ProceduralBlockSymbol &proc_symbol) {    
        fmt::print("visit module instance: {}\n", proc_symbol.name);
    }

[[maybe_unused]] void ModuleDefinitionVisitor::handle(const slang::InstanceSymbol &symbol) {
    fmt::print("visit module instance: {}\n", symbol.name);
    auto const &def = symbol.getDefinition();
    if (def.definitionKind == slang::DefinitionKind::Module) {
        modules.emplace(def.name, &symbol);
    }
    visitDefault(symbol);
}

DependencyAnalysisVisitor::Node *get_node_(DependencyAnalysisVisitor::Graph *graph,
                                           const slang::Symbol &sym) {
    auto n = std::string(sym.name);
    if (graph->node_mapping.find(n) == graph->node_mapping.end()) {
        auto ptr = std::make_unique<DependencyAnalysisVisitor::Node>(sym);
        auto &node = graph->nodes.emplace_back(std::move(ptr));
        graph->node_mapping.emplace(n, node.get());
    }
    return graph->node_mapping.at(n);
}

DependencyAnalysisVisitor::Node *DependencyAnalysisVisitor::Graph::get_node(
    const slang::NamedValueExpression *name) {
    auto const &sym = name->symbol;
    return get_node_(this, sym);
}

DependencyAnalysisVisitor::Node *DependencyAnalysisVisitor::Graph::get_node(
    const std::string &name) const {
    if (node_mapping.find(name) != node_mapping.end()) {
        return node_mapping.at(name);
    } else {
        return nullptr;
    }
}

DependencyAnalysisVisitor::Node *DependencyAnalysisVisitor::Graph::get_node(
    const slang::Symbol &symbol) {
    switch (symbol.kind) {
        case slang::SymbolKind::ProceduralBlock:
        case slang::SymbolKind::ContinuousAssign: {
            // procedural block doesn't have a name
            // we will only call it once since
            if (new_names_.find(&symbol) == new_names_.end()) {
                std::string name = ".blk{0}" + std::to_string(procedural_blk_count_++);
                new_names_.emplace(&symbol, name);
                auto &node = nodes.emplace_back(std::make_unique<Node>(symbol));
                node_mapping.emplace(name, node.get());
            }
            return node_mapping.at(new_names_.at(&symbol));
        }
        case slang::SymbolKind::Variable:
        case slang::SymbolKind::Net: {
            return get_node_(this, symbol);
        }
        default: {
            throw std::runtime_error(
                fmt::format("Unsupported node {0}", slang::toString(symbol.kind)));
        }
    }
}

DependencyAnalysisVisitor::DependencyAnalysisVisitor(const slang::Symbol *target)
    : target_(target) {
    graph_ = std::make_unique<Graph>();
    graph = graph_.get();
}

} // namespace svcirct
