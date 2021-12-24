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
#include "fmt/format.h"

#include <unordered_set>
#include <string>

namespace svcirct {
    
[[maybe_unused]] void ModuleDefinitionVisitor::handle(const slang::InstanceSymbol &symbol) {
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
