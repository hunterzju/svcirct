#ifndef _SVCIRCT_MLIRGEN_H
#define _SVCIRCT_MLIRGEN_H

#include <memory>

#include "slang/symbols/ASTVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/IR/Verifier.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace svcirct {

using namespace slang;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, const RootSymbol &root_symbol);
} // namespace svcirct 

#endif