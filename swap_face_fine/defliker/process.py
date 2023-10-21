import os


def defliker_cmd_pipeline(frames_folder: str,
                          fps: int = 25,
                          ):
    atlas_generation_cmd = "python src/stage1_neural_atlas.py --vid_name {}".format(frames_folder)
    neural_filter_and_refinement_cmd = "python src/neural_filter_and_refinement.py --video_name {}".format(frames_folder)

    os.system(atlas_generation_cmd)
    os.system(neural_filter_and_refinement_cmd)
