# Image_synth
Remember to check every default arguments of python files and change to your own!

# Environment
Device: Google Cloud Platfrom n1-standard-4
Python version: 3.6
Pytorch version: 1.04

# Process data
Place data in a folder "data" under "Project" where it contains all download subject folder from Box.

Run scripts (run in this order):
All the default arguments are set for whole dataset processing
1. rename_sentance.py
2. create_audio.py
3. create_obj.py
4. create_label.py
5. align_frame.py

# Training 
Specify all the arguments in train.py and run it.
The trained checkpints with a training arguments detail(info.txt) will be stored under "model".

# Testing
Simplely specify the folder name under "model" as --model_path and run it.
The result and a copy of info.txt will be stored under "result"

# Validation
To check the quilty of the NN output:
1. export the mesh videos using [Blender](https://www.blender.org/)
2. Specify the output video and label video as input in meshviwer.py and run it
