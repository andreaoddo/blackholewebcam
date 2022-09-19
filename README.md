# blackholewebcam
BlackHole WebCam is a short Python script that emulates the light distortion due to a black hole on the image captured live by a webcam.
The computation implemented in the code are from the paper by Rodriguex C., Marin C. A. 2017, https://arxiv.org/pdf/1701.04434.pdf
In order to work, the script needs to run in an environment where `numpy`, `scipy` and `opencv-python` are already installed.

# Usage
The script works through a command-line interface, where arguments can be specified as follows:
* `--cams` lists the ids of all available webcams; in this mode, the script exits automatically after listing the devices.
* `-r` sets the radius in pixels of the black hole; if unspecified, defaults to 20.
* `-i` sets the id of the webcam that is going to be used; if unspecified, defaults to 1.
