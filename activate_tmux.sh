SESSION="vscode `pwd | md5sum | cut -b -3`"
echo $SESSION
/home/dcast/anaconda3/envs/deep_learning_torch/bin/tmux attach-session -t $SESSION || /home/dcast/anaconda3/envs/deep_learning_torch/bin/tmux new -s $SESSION