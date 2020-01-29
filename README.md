# ChessBoardFinder
Finding a Chessboard on images

This is related to https://github.com/WolfgangFahl/play-chess-with-a-webcam please have a look into this project if you are interested in chess game analysis/recognition.

This is a try to detect a chessboard on an image or video file by using opencv and numpy but no machine learning libs. 

ChessBoardFinder.py can find a chessboard under the following conditions:
- there is exactly one chessboard on the image
- the camera that took the image is supposed to be directly above the chessboard (no angle between camera and board), however there are good chances this algorithm works for angles up to 30% in z-axis
- the chessboard is roughly centered on the image (the center of the board must be within the innermost 66% of the image area)
- the chessboard should not be already warped (filling the image completely to 100%)
- the image must not be too dark or overexposed 
- no artefacts in the image (bad jpg/mpeg quality)
- If there are chess pieces on the board they must be placed in correct start position

This algorithm searches for saddle points in the image and does not use opencv findChessboardCorners() method. 
The chessboard is found quite reliably and this algorithm is faster than findChessboardCorners().

How to use:

python ChessBoardFinder.py --debug video.avi

python ChessBoardFinder.py --debug video.avi --startframe 10

python ChessBoardFinder.py --debug --image pic.jpg

There are some example videos of chessgames at https://files.klaube.net (directory index is enabled) so you could also try:

python ChessBoardFinder.py https://files.klaube.net/TK_scholarsmate30.avi --debug --startframe 5
