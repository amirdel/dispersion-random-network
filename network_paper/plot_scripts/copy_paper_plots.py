import os
import shutil as sh

network_paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_plots_folder = os.path.join(network_paper_folder, 'temp_plots')
final_folder = os.path.join(network_paper_folder, 'plots')
# for each folder specify the source folder, name of files in source folder, name of files in paper folder
copy_array = []
# 2d, 3d comparison plots
compare_path = os.path.join(temp_plots_folder, '2d_3d_compare')
compare_source_names = ['bt_linear_75_zoom', 'msd_x_zoom', 'msd_y_zoom', 'spread_x_sigma_2_oneTime_320',
                       'xy_side']
compare_dest_names = ['bt_linear_75_zoom', 'msd_x_zoom', 'msd_y_zoom', 'spread_280_zoom', 'xy_side']
copy_array.append([compare_path, compare_source_names, compare_dest_names, '.pdf'])
# purturbed plots
purt_path = os.path.join(temp_plots_folder, 'purt')
source = ['bt_linear_75', 'xy_side_onTime_80', 'xy_side_onTime_290']
dest = ['purt20_bt_linear_75', 'purt20_xy_side_80', 'purt20_xy_side_290']
copy_array.append([purt_path, source, dest, '.pdf'])
# contour plots
contour_path = os.path.join(temp_plots_folder, 'planar')
source = ['2d_sigma_2_0', '2d_sigma_2_3']
dest = ['2d_20_0', '2d_20_3']
copy_array.append([contour_path, source, dest, '.png'])

for item in copy_array:
    inp_folder = item[0]
    ext = item[-1]
    for source_name, dest_name in zip(item[1], item[2]):
        source_add = os.path.join(inp_folder, source_name+ext)
        dest_add = os.path.join(final_folder, dest_name+ext)
        print source_add
        print dest_add
        print '----'
        sh.copyfile(source_add, dest_add)