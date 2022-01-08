llhd.proc @dff_sync_reset.param1.always_ff.53.1(%data: !llhd.sig<i1>, %clk: !llhd.sig<i1>, %reset: !llhd.sig<i1>) -> (%q: !llhd.sig<i1> ) {
    br ^init
^init:
    %clk.prb = llhd.prb %clk : !llhd.sig<i1>
    llhd.wait (%clk : !llhd.sig<i1>), ^check
^check:
    %clk.prb1 = llhd.prb %clk : !llhd.sig<i1>
    %0 = hw.constant 0 : i1
    %1 = comb.icmp "eq" %clk.prb, %0 : i1
    %2 = comb.icmp "ne" %clk.prb1, %0 : i1
    %posedge = comb.and %1, %2 : i1
    cond_br %posedge, ^event, ^init
^event:
    %reset.prb = llhd.prb %reset : !llhd.sig<i1>
    %3 = hw.constant -1 : i1
    %4 = comb.xor %3, %reset.prb : i1
    %5 = comb.icmp "ne" %4, %0 : i1
    %6 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    cond_br %5, ^if_true, ^if_false
^if_true:
    llhd.drv %q, %0 after %6 : !llhd.sig<i1>
    br ^init
^if_false:
    %data.prb = llhd.prb %data : !llhd.sig<i1>
    llhd.drv %q, %data.prb after %6 : !llhd.sig<i1>
    br ^init
}

llhd.entity @dff_sync_reset.param1(%data: !llhd.sig<i1>, %clk: !llhd.sig<i1>, %reset: !llhd.sig<i1>) -> (%q: !llhd.sig<i1> ) {
    llhd.inst "inst" @dff_sync_reset.param1.always_ff.53.1(%data, %clk, %reset) -> (%q) : (!llhd.sig<i1>, !llhd.sig<i1>, !llhd.sig<i1>) -> (!llhd.sig<i1>)
}

llhd.proc @tb_top.always.116.0() -> (%clk: !llhd.sig<i1> ) {
    br ^0
^0:
    %1 = llhd.prb %clk : !llhd.sig<i1>
    %clk.shadow = llhd.var %1 : i1
    %2 = hw.constant 10 : i32
    llhd.wait  for %2, ^3
^3:
    %4 = llhd.prb %clk : !llhd.sig<i1>
    llhd.store %clk.shadow, %4 : !llhd.ptr<i1>
    %5 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %clk.shadow.ld = llhd.load %clk.shadow : !llhd.ptr<i1>
    %6 = hw.constant -1 : i1
    %7 = comb.xor %6, %clk.shadow.ld : i1
    llhd.drv %clk, %7 after %5 : !llhd.sig<i1>
    br ^0
}

llhd.proc @tb_top.initial.199.0() -> (%reset: !llhd.sig<i1> , %d: !llhd.sig<i1> ) {
    br ^0
^0:
    %1 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %2 = hw.constant 0 : i1
    llhd.drv %reset, %2 after %1 : !llhd.sig<i1>
    llhd.drv %d, %2 after %1 : !llhd.sig<i1>
    %3 = hw.constant 10 : i32
    llhd.wait  for %3, ^4
^4:
    %5 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %6 = hw.constant 1 : i1
    llhd.drv %reset, %6 after %5 : !llhd.sig<i1>
    %7 = hw.constant 5 : i32
    llhd.wait  for %7, ^8
^8:
    %9 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %10 = hw.constant 1 : i1
    llhd.drv %d, %10 after %9 : !llhd.sig<i1>
    %11 = hw.constant 8 : i32
    llhd.wait  for %11, ^12
^12:
    %13 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %14 = hw.constant 0 : i1
    llhd.drv %d, %14 after %13 : !llhd.sig<i1>
    %15 = hw.constant 2 : i32
    llhd.wait  for %15, ^16
^16:
    %17 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %18 = hw.constant 1 : i1
    llhd.drv %d, %18 after %17 : !llhd.sig<i1>
    %19 = hw.constant 10 : i32
    llhd.wait  for %19, ^20
^20:
    %21 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    %22 = hw.constant 0 : i1
    llhd.drv %d, %22 after %21 : !llhd.sig<i1>
    llhd.halt
}

llhd.entity @tb_top() -> () {
    %0 = hw.constant 0 : i1
    %clk = llhd.sig "clk" %0 : i1
    %reset = llhd.sig "reset" %0 : i1
    %d = llhd.sig "d" %0 : i1
    %q = llhd.sig "q" %0 : i1
    llhd.inst "inst" @dff_sync_reset.param1(%d, %clk, %reset) -> (%q) : (!llhd.sig<i1>, !llhd.sig<i1>, !llhd.sig<i1>) -> (!llhd.sig<i1>)
    llhd.inst "inst1" @tb_top.always.116.0() -> (%clk) : () -> (!llhd.sig<i1>)
    llhd.inst "inst2" @tb_top.initial.199.0() -> (%reset, %d) : () -> (!llhd.sig<i1>, !llhd.sig<i1>)
}
