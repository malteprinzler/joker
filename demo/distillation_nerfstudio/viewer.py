#!/home/mprinzler/miniconda3/envs/nerfstudio/bin/python3.8
# -*- coding: utf-8 -*-
import re
import sys
import tyro
from nerfstudio.scripts.viewer.run_viewer import RunViewer as RunViewerOrig
from nerfstudio.scripts.viewer.run_viewer import _start_viewer
from nerfstudio.utils.eval_utils import eval_setup

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()

class RunViewer(RunViewerOrig):
    '''
    CHANGED test_mode='inference' compared to original RunViewer in nerfstudio.scripts.viewer.run_viewer so that doesnt have to load dataset images
    '''

    def main(self) -> None:
        """Main function."""
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="inference",  # CHANGED compared to original RunViewer in nerfstudio.scripts.viewer.run_viewer so that doesnt have to load dataset images
        )
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk
        config.viewer.max_num_display_images = 0 # CHANGED compared to original RunViewer in nerfstudio.scripts.viewer.run_viewer so that doesnt have to load dataset images

        _start_viewer(config, pipeline, step)


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(entrypoint())
