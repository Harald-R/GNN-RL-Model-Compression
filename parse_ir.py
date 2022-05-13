import re

def get_tensor_size(type_str):
    type_info = type_str.split('x')
    sizes = [int(dim) for dim in type_info[:-1]]
    elem_type = type_info[-1]
    return sizes, elem_type

def generate_op_info(op_info, op_name, op_type, operands, output_shape, output_elem_type):
    return {
        'op_info': op_info,
        'op_name': op_name,
        'op_type': op_type,
        'operands': operands,
        'output_shape': output_shape,
        'output_elem_type': output_elem_type
    }

def get_elem_type_byte_size(elem_type):
    elem_types = {
        'u8': 1,
        'ui8': 1,
        'i8': 1,
        'f16': 2,
        'f32': 4,
        'si32': 4
    }
    return elem_types[elem_type]

def get_op_type_encoding(op_type):
    op_types = {
        'VPU.NCE.Convolution' : 0,
        'VPU.NCE.DepthConvolution': 1,
        'VPU.NCE.MaxPool': 2,
        'VPU.NCE.Eltwise': 3
    }
    return op_types[op_type]

operations = {}
sw_operations = {}
constants = {}

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

        if op_type == 'const.Declare':
            constants[value] = generate_op_info(op_info, op_name, op_type, operands, output_shape, output_elem_type)
        else:
            if op_type.startswith('VPU.NCE'):
                operations[value] = generate_op_info(op_info, op_name, op_type, operands, output_shape, output_elem_type)
            else:
                sw_operations[value] = generate_op_info(op_info, op_name, op_type, operands, output_shape, output_elem_type)

for value, op_information in operations.items():
    op_type = op_information['op_type']
    operands = op_information['operands']

    if op_type in ['VPU.NCE.Convolution', 'VPU.NCE.DepthConvolution']:
        act_window = None
        if len(operands) == 3:
            input, weights, weights_table = operands
        elif len(operands) == 4:
            input, weights, weights_table, act_window = operands
        else:
            raise Exception('{} with wrong number of operands: {}'.format(op_type, len(operands)))

        if input in operations:
            input_shape, input_elem_type = operations[input]['output_shape'], operations[input]['output_elem_type']
        else:
            input_shape, input_elem_type = sw_operations[input]['output_shape'], sw_operations[input]['output_elem_type']
        weights_shape, weights_elem_type = constants[weights]['output_shape'], constants[weights]['output_elem_type']
        wt_shape, wt_elem_type = constants[weights_table]['output_shape'], constants[weights_table]['output_elem_type']
        if act_window:
            actwin_shape, actwin_elem_type = constants[act_window]['output_shape'], constants[act_window]['output_elem_type']
            operations[value]['act_win_shape'] = actwin_shape
            operations[value]['act_win_elem_type'] = actwin_elem_type

        operations[value]['input_shape'] = input_shape
        operations[value]['input_elem_type'] = input_elem_type
        operations[value]['weights_shape'] = weights_shape
        operations[value]['weights_elem_type'] = weights_elem_type
        operations[value]['wt_shape'] = wt_shape
        operations[value]['wt_elem_type'] = wt_elem_type

    elif op_type in ['VPU.NCE.MaxPool']:
        input, weights, act_window = operands

        if input in operations:
            input_shape, input_elem_type = operations[input]['output_shape'], operations[input]['output_elem_type']
        else:
            input_shape, input_elem_type = sw_operations[input]['output_shape'], sw_operations[input]['output_elem_type']
        wt_shape, wt_elem_type = constants[weights_table]['output_shape'], constants[weights_table]['output_elem_type']
        actwin_shape, actwin_elem_type = constants[act_window]['output_shape'], constants[act_window]['output_elem_type']

        operations[value]['input_shape'] = input_shape
        operations[value]['input_elem_type'] = input_elem_type
        operations[value]['wt_shape'] = wt_shape
        operations[value]['wt_elem_type'] = wt_elem_type
        operations[value]['act_win_shape'] = actwin_shape
        operations[value]['act_win_elem_type'] = actwin_elem_type
    
    elif op_type in ['VPU.NCE.Eltwise']:
        input1, input2 = operands

        if input1 in operations:
            input1_shape, input1_elem_type = operations[input1]['output_shape'], operations[input1]['output_elem_type']
        else:
            input1_shape, inpu1t_elem_type = sw_operations[input1]['output_shape'], sw_operations[input1]['output_elem_type']

        if input2 in operations:
            input2_shape, input2_elem_type = operations[input2]['output_shape'], operations[input2]['output_elem_type']
        else:
            input2_shape, input2_elem_type = sw_operations[input2]['output_shape'], sw_operations[input2]['output_elem_type']

        operations[value]['input1_shape'] = input1_shape
        operations[value]['input1_elem_type'] = input1_elem_type
        operations[value]['input2_shape'] = input2_shape
        operations[value]['input2_elem_type'] = input2_elem_type

for value, op_information in operations.items():
    operands_to_remove = []
    for operand in op_information['operands']:
        if operand in list(constants.keys()):
            operands_to_remove.append(operand)
        elif operand not in list(operations.keys()):
            operands_to_remove.append(operand)
    if operands_to_remove:
        for operand in operands_to_remove:
            op_information['operands'].remove(operand)

features = []
graph_op_indices = {}
graph_edges_source = []
graph_edges_destination = []

for value, op_information in operations.items():
    op_type = op_information['op_type']

    if op_type in ['VPU.NCE.Convolution', 'VPU.NCE.DepthConvolution']:
        _, input_C, input_H, input_W = op_information['input_shape']
        weights_OC, weights_IC, weights_KY, weights_KX = op_information['weights_shape']
        input_elem_byte_size = get_elem_type_byte_size(op_information['input_elem_type'])
        weights_elem_byte_size = get_elem_type_byte_size(op_information['weights_elem_type'])
    elif op_type in ['VPU.NCE.MaxPool']:
        _, input_C, input_H, input_W = op_information['input_shape']
        weights_OC, weights_IC, weights_KY, weights_KX = op_information['wt_shape']
        input_elem_byte_size = get_elem_type_byte_size(op_information['input_elem_type'])
        weights_elem_byte_size = get_elem_type_byte_size(op_information['wt_elem_type'])
    elif op_type in ['VPU.NCE.Eltwise']:
        _, input_C, input_H, input_W = op_information['input1_shape']
        weights_OC, weights_IC, weights_KY, weights_KX = op_information['input2_shape']
        input_elem_byte_size = get_elem_type_byte_size(op_information['input1_elem_type'])
        weights_elem_byte_size = get_elem_type_byte_size(op_information['input2_elem_type'])

    _, output_C, output_H, output_W = op_information['output_shape']
    output_elem_byte_size = get_elem_type_byte_size(op_information['output_elem_type'])

    op_type_encoded = get_op_type_encoding(op_type)

    # TODO: cover weights table and activation window
    op_features = [op_type_encoded,
                   input_C, input_H, input_W, input_elem_byte_size,
                   weights_OC, weights_IC, weights_KY, weights_KX, weights_elem_byte_size,
                   output_C, output_H, output_W, output_elem_byte_size]
    features.append(op_features)
    graph_op_indices[value] = len(features) - 1

    for operand in op_information['operands']:
        graph_edges_source.append(graph_op_indices[operand])
        graph_edges_destination.append(graph_op_indices[value])

print(features)
print(graph_op_indices)
print(graph_edges_source)
print(graph_edges_destination)