import gtsam
import graphviz
from collections import defaultdict


def display_factor_graph(factor_keyed:dict, variable_keyed:dict, SYMBOL_GAP = 10**6):

    dot = graphviz.Digraph(comment='Factor Graph', engine="neato")
    landmark_ages = defaultdict(lambda : 0)
    WIDTH = 4
    WIDTH_OFFSET = 2
    HEIGHT = 3
    HEIGHT_OFFSET = -3
    for x_var in variable_keyed:
        if x_var[0] == 'x':
            frame = int(x_var[1:])%SYMBOL_GAP
            dot.node(x_var, shape='circle', pos=f'{frame*WIDTH},0!', label=f"{x_var[0].upper()}_{int(x_var[1:])%SYMBOL_GAP}")
            print(x_var)
            for factor in variable_keyed[x_var]:
                if len(factor_keyed[factor]) == 1:  # unary camera prior factor
                    dot.node(factor, shape='box', pos=f'{frame*WIDTH},{-1}!', fillcolor='black', label='', style='filled', width='0.2',height='0.2')
                    dot.edge(x_var, factor, arrowhead='none')
                else:  # camera object between factor
                    for l_var in factor_keyed[factor]:
                        if l_var[0] == 'l':
                            landmark_ages[l_var] = frame
                            idx = int(l_var[1:])//SYMBOL_GAP
                            dot.node(l_var, shape='circle', pos=f'{frame*WIDTH + WIDTH_OFFSET},{idx*HEIGHT + HEIGHT_OFFSET + 2}!',label=f"{l_var[0].upper()}{idx}_{int(l_var[1:]) % SYMBOL_GAP}")
                            dot.node(factor, shape='box', pos=f'{frame*WIDTH + WIDTH_OFFSET/2},{idx*HEIGHT + HEIGHT_OFFSET + 1.5}!', fillcolor='red', label='',style='filled', width='0.2', height='0.2')
                            dot.edge(x_var, factor, arrowhead='none')
                            dot.edge(l_var, factor, arrowhead='none')
                            print(l_var)
    for triple_factor in factor_keyed:
        if len(factor_keyed[triple_factor]) == 3:
            vars = factor_keyed[triple_factor]
            x = (landmark_ages[vars[0]]*WIDTH + landmark_ages[vars[1]]*WIDTH + landmark_ages[vars[2]] + 2*WIDTH_OFFSET) / 2
            y = int(vars[0][1:])//SYMBOL_GAP
            dot.node(triple_factor, shape='box', pos=f'{x},{y*HEIGHT + HEIGHT_OFFSET + 2}!', fillcolor='blue', label='',style='filled', width='0.2', height='0.2')
            for var in factor_keyed[triple_factor]:
                dot.edge(var, triple_factor, arrowhead='none')
                if var[0] == 'v':
                    dot.node(var, shape='circle', pos=f'{x},{y*HEIGHT + HEIGHT_OFFSET + 3}!', label=f"{var[0].upper()}_{int(var[1:])%SYMBOL_GAP}")

    for velocity_between_factor in factor_keyed:
        if len(factor_keyed[velocity_between_factor]) == 2 and factor_keyed[velocity_between_factor][0][0] == 'v':
            vars = factor_keyed[velocity_between_factor]
            x = (landmark_ages['l' + vars[0][1:]] * WIDTH + landmark_ages['l' + vars[1][1:]] * WIDTH  + 2 * WIDTH_OFFSET - WIDTH) / 2
            y = int(vars[0][1:]) // SYMBOL_GAP
            dot.node(velocity_between_factor, shape='box', pos=f'{x},{y * HEIGHT + HEIGHT_OFFSET + 3}!', fillcolor='green', label='', style='filled', width='0.2', height='0.2')
            for var in factor_keyed[velocity_between_factor]:
                dot.edge(var, velocity_between_factor, arrowhead='none')

    dot.view()
    print('')
# def display_factor_graph_old(factor_keyed:dict, variable_keyed:dict, SYMBOL_GAP = 10**6):
#
#     dot = graphviz.Digraph(comment='Factor Graph', engine="neato")
#     landmark_ages = {}
#     for x_var in variable_keyed:
#         if x_var[0] == 'x':
#             frame = int(x_var[1:])%SYMBOL_GAP
#             # dot.node(x_var, shape='circle', pos=f'{frame*4},0!', label=f"{x_var[0].upper()}_{int(x_var[1:])%SYMBOL_GAP}")
#             print(x_var)
#             for factor in variable_keyed[x_var]:
#                 if len(factor_keyed[factor]) == 1:  # unary camera prior factor
#                     dot.node(factor, shape='box', pos=f'{frame*4},{-1}!', fillcolor='black', label='', style='filled', width='0.2',height='0.2')
#                     dot.edge(x_var, factor, arrowhead='none')
#                 else:  # camera object between factor
#                     for l_var in factor_keyed[factor]:
#                         if l_var[0] == 'l':
#                             idx = int(l_var[1:])//SYMBOL_GAP
#                             dot.node(l_var, shape='circle', pos=f'{frame*4 + 2},{idx*2 + 2}!',label=f"{l_var[0].upper()}{idx}_{int(l_var[1:]) % SYMBOL_GAP}")
#                             dot.node(factor, shape='box', pos=f'{frame*4 + 1},{idx*2 + 2}!', fillcolor='black', label='',style='filled', width='0.2', height='0.2')
#                             dot.edge(x_var, factor, arrowhead='none')
#                             dot.edge(l_var, factor, arrowhead='none')
#                             print(l_var)
#     for triple_factor in factor_keyed:
#         if len(factor_keyed[triple_factor]) == 3:
#             dot.node(triple_factor, shape='box', pos=f'{frame*4},0!', fillcolor='black', label='',style='filled', width='0.2', height='0.2')
#             for var in factor_keyed[triple_factor]:
#                 dot.edge(var, triple_factor, arrowhead='none')
#     dot.view()
#     print('')
