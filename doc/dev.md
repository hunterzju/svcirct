# 开发日志
## TODO
* [] 从slang的ast转换到circt中的llhd
    - Type定义
    - Op定义

## 当前进展
* 整合了slang工程和mlir工程

## 参考资料
systemverilog示例：
* https://www.chipverify.com/systemverilog/systemverilog-tutorial
* http://www.asic-world.com/examples/systemverilog/index.html

## 实现~~计划~~
~~参考`tests/benchmark`中`counter`实现`systemverilog`到`llhd`转换~~
算了，换个简单电路：`tests/benchmark/examples`中的`d_ff.sv`为基础；
### 层次结构
```bash
# d_ff.sv
# 模块定义
Module
    |- ModuleHeader
    |- AlwaysFFBlock
```

### module
module语法结构
```
Module definitions
    - Module header definition
    - Port declarations
    - Parameterized modules
    - Module contents

module_declaration ::= // from A.1.2
module_nonansi_header [ timeunits_declaration ] { module_item }
endmodule [ : module_identifier ]
| module_ansi_header [ timeunits_declaration ] { non_port_module_item }
endmodule [ : module_identifier ]
| { attribute_instance } module_keyword [ lifetime ] module_identifier ( .* ) ;
[ timeunits_declaration ] { module_item } endmodule [ : module_identifier ]
| extern module_nonansi_header
| extern module_ansi_header
module_nonansi_header ::=
{ attribute_instance } module_keyword [ lifetime ] module_identifier
{ package_import_declaration } [ parameter_port_list ] list_of_ports ;
module_ansi_header ::=
{ attribute_instance } module_keyword [ lifetime ] module_identifier
{ package_import_declaration } 1 [ parameter_port_list ] [ list_of_port_declarations ] ;
module_keyword ::= module | macromodule
timeunits_declaration ::=
timeunit time_literal [ / time_literal ] ;
| timeprecision time_literal ;
| timeunit time_literal ; timeprecision time_literal ;
| timeprecision time_literal ; timeunit time_literal ;
parameter_port_list ::= // from A.1.3
# ( list_of_param_assignments { , parameter_port_declaration } )
| # ( parameter_port_declaration { , parameter_port_declaration } )
| #( )
parameter_port_declaration ::=
parameter_declaration
| local_parameter_declaration
| data_type list_of_param_assignments
| type list_of_type_assignments
```