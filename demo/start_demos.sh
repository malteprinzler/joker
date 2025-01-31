#!/usr/bin/zsh
SESSION=joker_demos

tmux -2 new-session -d -s $SESSION

tmux split-window -v
tmux split-window -v

tmux select-pane -t 0
tmux send-keys 'conda activate joker; python demo/2d_prior_demo/app.py' C-m

tmux select-pane -t 1
tmux send-keys 'conda activate nerfstudio; python demo/distillation_nerfstudio/viewer.py --load-config assets/joker/nerfstudio_ckpts/000/nerfacto/2025-01-31_110250/config.yml' C-m

tmux select-pane -t 2
tmux send-keys 'nvtop' C-m

# Attach to session
tmux -2 attach-session -t $SESSION
