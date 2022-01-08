//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_ProcOp : Standalone_Op<"proc", []> {
  let summary = "Create a process";
  let description = [{
    A `llhd.proc` represents control-flow in a timed fashion. It allows a
    procedural description of how a circuit's output signals change in
    reaction to changing input signals. It has a region with arbitrarily
    many basic blocks. The first block is the entry block and cannot be
    targeted by the terminators. It uses `llhd.wait` as a terminator to add
    timed control-flow. Immediate control-flow with `br` or `cond_br` is
    also possible. Every process must either contain an infinite loop or
    terminate with the `llhd.halt` terminator.

    How does a process compare to functions and entities?

    | Unit     | Paradigm     | Timing    | Models                         |
    |----------|--------------|-----------|--------------------------------|
    | Function | control-flow | immediate | Computation in zero time       |
    | Process  | control-flow | timed     | Behavioral circuit description |
    | Entity   | data-flow    | timed     | Structural circuit description |

    Syntax:

    ```
    proc-op ::= `llhd.proc` proc-symbol `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `{` proc-region `}`
    ```

    Examples:

    ```mlir
    llhd.proc @example(%in0 : !llhd.sig<i64>, %in1 : !llhd.sig<i1>) ->
        (%out2 : !llhd.sig<i1>) {
      br ^bb1
    ^bb1:
      llhd.halt
    }
    ```
  }];
}

def Standalone_EntityOp : Standalone_Op<"entity", []> {
    let summary = "Create an entity";
    let description = [{
    from: llhd.entity
    The `llhd.entity` operation defines a new entity unit. An entity
    represents the data-flow description of how a circuit's output values
    change in reaction to changing input values.
    An entity contains one region with a single block and an implicit
    `TerminatorOp` terminator. Both the block name and terminator are
    omitted in the custom syntax. No further blocks and control-flow are
    legal inside an entity.

    Syntax:

    ```
    entity-op ::= `llhd.entity` entity-symbol `(` arg-list `)` `->`
      `(` out-list `)` attr-dict entity-region
    ```

    Example:

    ```mlir
    llhd.entity @Foo () -> () {
      %0 = llhd.const 0 : i1
      %toggle = llhd.sig %0 : i1 -> !llhd.sig<i1>
      %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
      %2 = llhd.not %1 : i1
      %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
      llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
    }
    ```
    }];
}

def Standalone_InstOp : Standalone_Op<"inst", []> {
  let summary = "Instantiates a process or entity.";
  let description = [{
    from llhd.inst
    Instantiates a process or entity and thus allows to build hierarchies.
    Can only be used within an entity. An instance defines a unique name
    within the entity it resides in.

    Syntax:

    ```
    inst-op ::= `llhd.inst` inst-name symbol-name `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `:`
      functional-type(ssa-input-list, ssa-output-list)
    ```

    Example:

    ```mlir
    llhd.inst "foo" @empty() -> () : () -> ()
    llhd.inst "bar" @proc_symbol() -> (%out0) : () -> !llhd.sig<i32>
    llhd.inst "baz" @entity_symbol(%in0, %in1) -> (%out0, %out1) :
      (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
    ```
  }];
}


#endif // STANDALONE_OPS
