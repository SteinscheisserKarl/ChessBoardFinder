# ChessBoardFinder
Finding a (real!) chessboard on images or videos

This is a try to detect a chessboard on an image or video file by using opencv and numpy but no machine learning libs. This has nothing
to do with opencv's camera calibration method. It is about real chessboards with chess pieces placed on the board.

ChessBoardFinder.py can find a chessboard under the following conditions:
- there is exactly one chessboard on the image
- the camera that took the image is supposed to be directly above the chessboard, however there are good chances this algorithm works if the chessboard is shifted along the X and/or Y axis and the X-Y plane (surface plane) is tilted around the X and/or Y axis
- the chessboard is roughly centered on the image (the center of the board must be within the innermost 66% of the image area)
- the chessboard should not be already warped (filling the image completely to 100%)
- the image must not be too dark or overexposed 
- no artefacts in the image (bad jpg/mpeg quality)
- If there are chess pieces on the board they must be placed in correct start position

This algorithm searches for saddle points in the image and does not use opencv findChessboardCorners() method (findChessboardCorners() is used for camera calibration and needs a "perfect" black/white chessboard like image for the calibration to be successful. This method is not suited to find real chessboard which are used by chess players in real life/over the board chess games).
The chessboard is found quite reliably and this algorithm is faster than findChessboardCorners().

How to use:

python ChessBoardFinder.py video.mp4 --debug

python ChessBoardFinder.py video.mp4 --debug --startframe 10

python ChessBoardFinder.py pic.jpg --debug --image

If you have not calibrated your camera you should try (slower but has more chances to find the board):</br>
python ChessBoardFinder.py video.mp4 --debug --noundistort --speedup 0 

If you want some padding pixels around the chessboard:</br>
python ChessBoardFinder.py video.mp4 --debug  --padding 10 

There are some example videos of chessgames at https://files.klaube.net (directory index is enabled) so you could also try:

python ChessBoardFinder.py https://files.klaube.net/chessgame_30.01.2020T2205.mp4 --debug --startframe 5 --noundistort --speedup 0
