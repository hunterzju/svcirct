#include "MLIRGen.h"

namespace {
using namespace slang;

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    /// Public API: convert the AST for a slang module (source file) to an MLIR
    /// Module operation.
    mlir::ModuleOp mlirGen(InstanceSymbol &inst_symbol) {
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
};

} // namespace

namespace svcirct
{
// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              InstanceSymbol &inst_symbol) {
  return MLIRGenImpl(context).mlirGen(inst_symbol);
}    

} // namespace svcirct
