llhd.proc @up_down_counter.param1.always.199.1(%clk: !llhd.sig<i1>, %reset: !llhd.sig<i1>, %up_down: !llhd.sig<i1>) -> (%counter_up_down: !llhd.sig<i4> ) {
    br ^0
^0:
    %1 = llhd.prb %counter_up_down : !llhd.sig<i4>
    %counter_up_down.shadow = llhd.var %1 : i4
    br ^init
^init:
    %clk.prb = llhd.prb %clk : !llhd.sig<i1>
    %reset.prb = llhd.prb %reset : !llhd.sig<i1>
    llhd.wait (%clk, %reset : !llhd.sig<i1>, !llhd.sig<i1>), ^check
^check:
    %2 = llhd.prb %counter_up_down : !llhd.sig<i4>
    llhd.store %counter_up_down.shadow, %2 : !llhd.ptr<i4>
    %clk.prb1 = llhd.prb %clk : !llhd.sig<i1>
    %3 = hw.constant 0 : i1
    %4 = comb.icmp "eq" %clk.prb, %3 : i1
    %5 = comb.icmp "ne" %clk.prb1, %3 : i1
    %posedge = comb.and %4, %5 : i1
    %reset.prb1 = llhd.prb %reset : !llhd.sig<i1>
    %6 = comb.icmp "eq" %reset.prb, %3 : i1
    %7 = comb.icmp "ne" %reset.prb1, %3 : i1
    %posedge1 = comb.and %6, %7 : i1
    %event_or = comb.or %posedge, %posedge1 : i1
    cond_br %event_or, ^event, ^init
^event:
    %8 = llhd.constant_time #llhd.time<0s, 1d, 0e>
    cond_br %7, ^if_true, ^if_false
^if_true:
    %9 = hw.constant 0 : i4
    llhd.drv %counter_up_down, %9 after %8 : !llhd.sig<i4>
    br ^0
^if_false:
    %up_down.prb = llhd.prb %up_down : !llhd.sig<i1>
    %10 = hw.constant -1 : i1
    %11 = comb.xor %10, %up_down.prb : i1
    %12 = comb.icmp "ne" %11, %3 : i1
    %13 = hw.constant 1 : i4
    cond_br %12, ^if_true1, ^if_false1
^if_true1:
    %counter_up_down.shadow.ld = llhd.load %counter_up_down.shadow : !llhd.ptr<i4>
    %14 = comb.add %counter_up_down.shadow.ld, %13 : i4
    llhd.drv %counter_up_down, %14 after %8 : !llhd.sig<i4>
    br ^0
^if_false1:
    %counter_up_down.shadow.ld1 = llhd.load %counter_up_down.shadow : !llhd.ptr<i4>
    %15 = comb.sub %counter_up_down.shadow.ld1, %13 : i4
    llhd.drv %counter_up_down, %15 after %8 : !llhd.sig<i4>
    br ^0
}

llhd.entity @up_down_counter.param1(%clk: !llhd.sig<i1>, %reset: !llhd.sig<i1>, %up_down: !llhd.sig<i1>) -> (%counter: !llhd.sig<i4> ) {
    %0 = hw.constant 0 : i4
    %counter_up_down = llhd.sig "counter_up_down" %0 : i4
    %1 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    %counter_up_down.prb = llhd.prb %counter_up_down : !llhd.sig<i4>
    llhd.drv %counter, %counter_up_down.prb after %1 : !llhd.sig<i4>
    llhd.inst "inst" @up_down_counter.param1.always.199.1(%clk, %reset, %up_down) -> (%counter_up_down) : (!llhd.sig<i1>, !llhd.sig<i1>, !llhd.sig<i1>) -> (!llhd.sig<i4>)
}

llhd.proc @updowncounter_tb.initial.63.0() -> (%clk: !llhd.sig<i1> ) {
    br ^0
^0:
    %1 = llhd.prb %clk : !llhd.sig<i1>
    %clk.shadow = llhd.var %1 : i1
    %2 = hw.constant 0 : i1
    %3 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    llhd.drv %clk, %2 after %3 : !llhd.sig<i1>
    llhd.store %clk.shadow, %2 : !llhd.ptr<i1>
    br ^loop_body
^loop_body:
    %4 = llhd.constant_time #llhd.time<5ns, 0d, 0e>
    llhd.wait  for %4, ^5
^5:
    %6 = llhd.prb %clk : !llhd.sig<i1>
    llhd.store %clk.shadow, %6 : !llhd.ptr<i1>
    %clk.shadow.ld = llhd.load %clk.shadow : !llhd.ptr<i1>
    %7 = hw.constant -1 : i1
    %8 = comb.xor %7, %clk.shadow.ld : i1
    %9 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    llhd.drv %clk, %8 after %9 : !llhd.sig<i1>
    llhd.store %clk.shadow, %8 : !llhd.ptr<i1>
    br ^loop_body
}

llhd.proc @updowncounter_tb.initial.112.0() -> (%reset: !llhd.sig<i1> , %up_down: !llhd.sig<i1> ) {
    br ^0
^0:
    %1 = hw.constant 1 : i1
    %2 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    llhd.drv %reset, %1 after %2 : !llhd.sig<i1>
    %3 = hw.constant 0 : i1
    llhd.drv %up_down, %3 after %2 : !llhd.sig<i1>
    %4 = llhd.constant_time #llhd.time<20ns, 0d, 0e>
    llhd.wait  for %4, ^5
^5:
    %6 = hw.constant 0 : i1
    %7 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    llhd.drv %reset, %6 after %7 : !llhd.sig<i1>
    %8 = llhd.constant_time #llhd.time<200ns, 0d, 0e>
    llhd.wait  for %8, ^9
^9:
    %10 = hw.constant 1 : i1
    %11 = llhd.constant_time #llhd.time<0s, 0d, 1e>
    llhd.drv %up_down, %10 after %11 : !llhd.sig<i1>
    llhd.halt
}

llhd.entity @updowncounter_tb() -> () {
    %0 = hw.constant 0 : i1
    %clk = llhd.sig "clk" %0 : i1
    %reset = llhd.sig "reset" %0 : i1
    %up_down = llhd.sig "up_down" %0 : i1
    %1 = hw.constant 0 : i4
    %counter = llhd.sig "counter" %1 : i4
    llhd.inst "inst" @up_down_counter.param1(%clk, %reset, %up_down) -> (%counter) : (!llhd.sig<i1>, !llhd.sig<i1>, !llhd.sig<i1>) -> (!llhd.sig<i4>)
    llhd.inst "inst1" @updowncounter_tb.initial.63.0() -> (%clk) : () -> (!llhd.sig<i1>)
    llhd.inst "inst2" @updowncounter_tb.initial.112.0() -> (%reset, %up_down) : () -> (!llhd.sig<i1>, !llhd.sig<i1>)
}
