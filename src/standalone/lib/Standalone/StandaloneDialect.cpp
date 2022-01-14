//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

using namespace mlir;
using namespace mlir::standalone;

#include "Standalone/StandaloneOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
}

// //===----------------------------------------------------------------------===//
// // Standalone dialect type.
// //===----------------------------------------------------------------------===//
// #include "llvm/ADT/TypeSwitch.h"
// #define GET_TYPEDEF_CLASSES
// #include "Standalone/StandaloneOpsTypes.h.inc"
// #define GET_TYPEDEF_CLASSES
// #include "Standalone/StandaloneOpsTypes.cpp.inc"

// Type StandaloneDialect::parseType(DialectAsmParser& parser) const {
//   llvm::StringRef mnemonic;
//   Type type;
//   // if (generatedTypeParser(parser, mnemonic, type).hasValue())
//   //   return type;
  
//   return type;
// }

// void StandaloneDialect::printType(Type type, DialectAsmPrinter &os) const {
//   ;
// }