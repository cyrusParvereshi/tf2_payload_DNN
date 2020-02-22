# tf2_payload_DNN
Training a neural network to follow the Payload tracks in Team Fortress 2

Hi, I watched Harrison Kinsley's tutorial on self-driving cars within GTA5: https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/
I was inspired to implement something similar in Team Fortress 2 with the payload, a special mode which involves pushing a cart 
from one side of the map to another. The cart stays between these railroad tracks, and I'm attempting to make a character follow the train tracks
and stay within them until they reach the end of the map. Using what I learned in the tutorial, I'm taking in pixel data from a minimized 
window of tf2 and feeding it to a neural network model, which is a keras implementation of AlexNet (Harrison also uses this in his tutorial).
Anyone interested in learning more should watch the tutorial, which is also available on Harrison's YouTube channel, https://www.youtube.com/user/sentdex.
