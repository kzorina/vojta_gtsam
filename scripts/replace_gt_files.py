from pathlib import Path
import shutil
dataset_folder = Path('/home/ros/kzorina/vojtas/ycbv/test')
new_data_folder = Path('/home/ros/kzorina/vojtas/ycbv/test_copy_json_out')
# dataset_folder = Path('/home/kzorina/work/bop_datasets/ycbv/test')
# new_data_folder = Path('/home/kzorina/work/bop_datasets/ycbv/test_copy_json_out')

scene_ids = list(range(48, 60))
for scene_id in scene_ids:
    for fname in ['scene_camera.json', 'scene_gt.json']:
        initial_path = dataset_folder / f'0000{scene_id}' / fname
        backup_path = dataset_folder / f'0000{scene_id}' / str(fname.split('.')[0] + '_backup.json')
        copy_from_path = new_data_folder / f'0000{scene_id}' / fname
        shutil.copy2(initial_path, backup_path)
        shutil.copy2(copy_from_path, initial_path)
