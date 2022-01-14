//===----------------------------------------------------------------------===//
// Basic data type implementations for the standalone dialect.
//===----------------------------------------------------------------------===//
/// FIXME: Implement Type system. Type need manually define parser and printer?
class StandaloneType<string name> : TypeDef<Standalone_Dialect, name> { }

def SigTypeImpl : StandaloneType<"Sig"> {
  let summary = "signal type";
  let description = [{
    Signals correspond directly to wires in a physical design, and are used to
    model propagation delays and timing. Signals are used to carry values across
    time steps in the LLHD execution model.
  }];

  let mnemonic = "sig";
  // let parameters = (ins "::mlir::Type":$underlyingType);
  // let assemblyFormat = "`<` $underlyingType `>`";

  // let builders = [
  //   AttrBuilderWithInferredContext<(ins "::mlir::Type":$underlyingType), [{
  //     return $_get(underlyingType.getContext(), underlyingType);
  //   }]>
  // ];
}