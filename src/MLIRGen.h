#ifndef _SVCIRCT_MLIRGEN_H
#define _SVCIRCT_MLIRGEN_H_

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace svcirct {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace toy

#endif