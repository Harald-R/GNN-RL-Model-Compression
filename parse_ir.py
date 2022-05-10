import re

operations = []

with open('resnet50_files/resnet50_ir_before_pass.mlir') as f:
    lines = f.readlines()
    for line in lines:
        p = re.compile('(%[^ ]+) = (.+) loc\("([^"]+)"\)')
        result = p.search(line)
        if not result:
            continue
        return_value, op_info, op_name = result.groups()
        operations.append([return_value, op_info, op_name])

print(operations)