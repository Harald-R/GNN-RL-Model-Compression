import re

def get_tensor_size(type_str):
    type_info = type_str.split('x')
    sizes = [int(dim) for dim in type_info[:-1]]
    elem_type = type_info[-1]
    return sizes, elem_type

operations = []

with open('resnet50_files/resnet50_ir_before_pass.mlir') as f:
    lines = f.readlines()
    for line in lines:
        p = re.compile('(%[^ ]+) = (.+) loc\("([^"]+)"\)')
        result = p.search(line)
        if not result:
            continue
        value, op_info, op_name = result.groups()

        op_type = op_info.split(' ')[0]
        op_type = op_type.split('(')[0]

        operands = []
        if op_type in ['VPU.NCE.Convolution', 'VPU.NCE.DepthConvolution', 'VPU.NCE.MaxPool', 'VPU.NCE.Eltwise', 'IE.PermuteCast', 'IE.AffineReshape', 'IE.Convert', 'IE.Negative', 'IE.ScaleShift']:
            operands = op_info.split('(')[1]
            operands = operands.split(')')[0]
            operands = operands.split(',')
            operands = [operand.strip() for operand in operands]

            return_type = ''.join(op_info.split('-> tensor<')[1:])
            return_type = return_type.split(',')[0]
            return_type = return_type.split('>')[0]
            output_shape, output_elem_type = get_tensor_size(return_type)
        elif op_type == 'IE.Slice':
            operands = op_info.split(' ')[1]

            return_type = ''.join(op_info.split(':')[1:]).strip()
            return_type = ''.join(op_info.split('tensor<')[1:])
            return_type = return_type.split(',')[0]
            return_type = return_type.split('>')[0]
            output_shape, output_elem_type = get_tensor_size(return_type)
        elif op_type == 'const.Declare':
            return_type = ''.join(op_info.split('tensor<')[1:])
            return_type = return_type.split(',')[0]
            return_type = return_type.split('>')[0]
            output_shape, output_elem_type = get_tensor_size(return_type)
        else:
            print('Unsupported op:', op_type)
            continue

        operations.append([value, op_info, operands, output_shape, output_elem_type, op_name])

print(operations)