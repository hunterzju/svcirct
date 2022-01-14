#include "MLIRGen.h"

#include "llvm/ADT/ScopedHashTable.h"
namespace {
using namespace slang;

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    /// Public API: convert the AST for a slang module (source file) to an MLIR
    /// Module operation.
    mlir::ModuleOp mlirGen(const RootSymbol &root_symbol) {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        // for (FunctionAST &F : moduleAST) {
        //   auto func = mlirGen(F);
        //   if (!func)
        //     return nullptr;
        //   theModule.push_back(func);
        // }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the Toy operations.
        if (failed(mlir::verify(theModule))) {
          theModule.emitError("module verification error");
          return nullptr;
        }

        return theModule;
    }
private:
    /// A "module" matches a Toy source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments are
    /// added to the mapping. When the processing of a function is terminated, the
    /// scope is destroyed and the mappings created in this scope are dropped.
    llvm::ScopedHashTable<mlir::StringRef, mlir::Value> symbolTable;

    mlir::FuncOp mlirGen(InstanceSymbol &inst) {
      // This is a generic function, the return type will be inferred later.
      // Arguments type are uniformly unranked tensors.
      mlir::Location location = builder.getUnknownLoc();
      mlir::FunctionType func_type = nullptr;
      return mlir::FuncOp::create(location, inst.name, func_type);
  }
};

} // namespace

namespace svcirct
{
  // The public API for codegen.
  mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, const RootSymbol &root_symbol) {
  return MLIRGenImpl(context).mlirGen(root_symbol);
}    

} // namespace svcirct
