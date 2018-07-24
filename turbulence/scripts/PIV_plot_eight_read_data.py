import library.tools.rw_data as rw
import library.display.graph as graph



basedir = "/Volumes/labshared3-1/takumi/2018_05_13_mid/freq1Hz/PIV_AnalysisResults/PIV_sv_vp_left_macro55mm_fps2000_D25p6mm_piston13p1mm_freq1Hz_v100mms_fx0p1259mmpx_setting1/time_evolution/"

marker = ['v', '^', 's', 'o']  # makers corresponding to domains
fillstyle = ['full', 'none']  # Positive core: fill, Negative core: no fill
colors = graph.get_first_n_colors_from_color_cycle(4)

for domain_num in range(1, 5):
    filename = "time_evolution_data_domain{0}_before_collision.json".format(domain_num)
    filepath = basedir + filename
    data = rw.read_json(filepath)

    cx_max, cy_max = data["cxmaxlist"], data["cymaxlist"]
    cx_min, cy_min = data["cxminlist"], data["cyminlist"]
    core_positions = [[cx_max, cy_max],[cx_min, cy_min]]
    fig, ax = graph.set_fig(fignum=3, subplot=111, figsize=(16, 10))
    for j, core_position in enumerate(core_positions):
        cx, cy = core_position[0], core_position[1]
        ax.plot(cx, cy, color=colors[domain_num - 1], marker=marker[domain_num - 1],
                  alpha=0.7, fillstyle=fillstyle[j])
        graph.setaxes(ax, 0,127, 0, 90)
        graph.labelaxes(ax, 'x [mm]', 'y [mm]')
        ax.invert_yaxis()
filename = 'trajectories_1'
graph.save(basedir + filename)
graph.show()