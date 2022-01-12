//===- StandaloneDialect.td - Standalone dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Standalone_Dialect : Dialect {
    let name = "standalone";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::standalone";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Standalone_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Standalone_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Basic data type implementations for the standalone dialect.
//===----------------------------------------------------------------------===//
class StandaloneType<string name> : TypeDef<Standalone_Dialect, name> { }

def SigTypeImpl : StandaloneType<"Sig"> {
  let summary = "signal type";
  let description = [{
    Signals correspond directly to wires in a physical design, and are used to
    model propagation delays and timing. Signals are used to carry values across
    time steps in the LLHD execution model.
  }];

  let mnemonic = "sig";
  let parameters = (ins "::mlir::Type":$underlyingType);
  let assemblyFormat = "`<` $underlyingType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$underlyingType), [{
      return $_get(underlyingType.getContext(), underlyingType);
    }]>
  ];
}

#endif // STANDALONE_DIALECT
