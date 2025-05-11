#!/bin/bash
for binary in ./bin/*; do
    filename=$(basename "$binary")
    
    echo "正在处理: $filename"
    llvm-bolt --print-disasm "$binary" > "./disasm/${filename}_disasm_output.txt" 2>&1 -o "./bolt_output/${filename}.bolt"
done

echo "所有文件处理完成"