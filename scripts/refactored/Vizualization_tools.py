import gtsam
import graphviz
from collections import defaultdict
from ScenePlotter import Plotter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from State import State, BareTrack
import numpy as np

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

def animate_refinement(refined_scene, scene_gt=None, scene_camera=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotter = Plotter(ax)

    def update_view(val):
        plotter.reset_default_lim()
        num = int(slider.val)
        plotter.clear()
        for obj_label in refined_scene[num]:
            for obj_idx in range(len(refined_scene[num][obj_label])):
                track = refined_scene[num][obj_label][obj_idx]
                if track["valid"]:
                # if True:
                    T_co:gtsam.Pose3 = gtsam.Pose3(track['T_co'])
                    T_wc:gtsam.Pose3 = gtsam.Pose3(track['T_wc'])
                    T_wo:gtsam.Pose3 = T_wc * T_co
                    Q_w = track['Q'] # covariance in the reference frame of the object
                    # R = T_wo.matrix()[:3, :3]
                    # Q_w = R @ Q_o @ R.T
                    plotter.plot_Q(Q_w[3:6, 3:6]*1000, T_wo)
                    plotter.plot_Q(Q_w[:3, :3]*10, T_wo, color='orange')
                    plotter.plot_T(T_wo)
        if scene_gt is not None:
            for obj_label in scene_gt[num]:
                for obj_idx in range(len(scene_gt[num][obj_label])):
                    T_cw = gtsam.Pose3(scene_camera[num]['T_cw'])
                    T_co = gtsam.Pose3(scene_gt[num][obj_label][obj_idx])
                    T_wo = T_cw.inverse() * T_co
                    plotter.plot_T(T_wo, alpha=0.3, size=0.3)
        for i in range(max(0, num-10), num + 1):
            for obj_label in refined_scene[i]:
                track = refined_scene[i][obj_label][0]
                T_wc = gtsam.Pose3(track['T_wc'])
                plotter.plot_T(T_wc)
                break


    axhauteur = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(axhauteur, 'frame', 0, len(refined_scene) - 1, valinit=0)
    slider.on_changed(update_view)
    update_view(None)
    # ani = animation.FuncAnimation(fig, update_view, len(refined_scene), fargs=(refined_scene, plotter), interval=100)


    # plotter.set_camera_view()
    plt.show()


def animate_state(state, initial_time_stamp, white_list=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotter = Plotter(ax)

    def update_view(val):
        plotter.reset_default_lim()
        num = slider.val
        plotter.clear()
        time_stamp = initial_time_stamp + num
        for obj_label in state.bare_tracks:
            if white_list is not None:
                if obj_label in white_list:
                    for obj_idx in range(len(state.bare_tracks[obj_label])):
                        bare_track:BareTrack = state.bare_tracks[obj_label][obj_idx]
                        T_wo, Q_w = bare_track.extrapolate(time_stamp)

                        if np.linalg.det(Q_w[3:6, 3:6])**(1/3) < 0.1 and np.linalg.det(Q_w[:3, :3])**(1/3) < 1:
                            plotter.plot_Q(Q_w[3:6, 3:6]*100, T_wo)
                            plotter.plot_Q(Q_w[:3, :3]*1, T_wo, color='orange')
                            plotter.plot_T(T_wo)
                            plotter.plot_T(T_wo, alpha=0.3, size=0.3)

    axhauteur = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(axhauteur, 'dt', 0, 1, valinit=0)
    slider.on_changed(update_view)
    update_view(None)
    # ani = animation.FuncAnimation(fig, update_view, len(refined_scene), fargs=(refined_scene, plotter), interval=100)

    # plotter.set_camera_view()
    plt.show()

if __name__ == "__main__":
    pass