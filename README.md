
## tf_opencv
C++ project, which use Tensorflow DLL and use Opencv DLL to do predict and train.

This DEMO project runs a vision task on Video from Elevator Security Camera. And to count the head and get elevator's running information. 

For head counting, this project use a pre-trained frozen PB file which use the Keras-Faster-RCNN network, and head rect data from MPII dataset.
For floor and arrow information, this project use a semi-manually data and a simple 2-layer Nerual Network trained frozen PB file.

## tensorflow dll and library downloads

- tensorflow.dll link:https://pan.baidu.com/s/1jIBriPw pwd:b4ay

- tensorflow.lib link: https://pan.baidu.com/s/1eS2BSQ2 pwd: t2a9

- libprotobuf.lib link: https://pan.baidu.com/s/1kVMK3Uv pwd: kbfv

## opencv dll

## fixed debug build, so that can debug the code

Task List

- [ ] feed network with two/more inputs (NO UI)

- [ ] get two/more output from predict (NO UI)

- [ ] image process: get a specified rect from image to a Tensorflow Tensor

- [X] load RPN network and predict (NO UI)

- [ ] implement helper to get ROI as input for classifier network input (NO UI)


