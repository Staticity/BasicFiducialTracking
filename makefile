CC          = g++
CFLAGS      = -Wall -c $(DEBUG) -O3
LFLAGS      = -Wall $(DEBUG)
INCLUDE_DIR = -I/usr/local/include/opencv -I/usr/local/include/opencv2
LIBRARY_DIR = -L/usr/local/lib
LIBRARIES   = -lopencv_calib3d     \
              -lopencv_core        \
              -lopencv_features2d  \
              -lopencv_flann       \
              -lopencv_highgui     \
              -lopencv_imgcodecs   \
              -lopencv_imgproc     \
              -lopencv_ml          \
              -lopencv_objdetect   \
              -lopencv_photo       \
              -lopencv_shape       \
              -lopencv_stitching   \
              -lopencv_superres    \
              -lopencv_ts          \
              -lopencv_video       \
              -lopencv_videoio     \
              -lopencv_videostab   \
              -lopencv_xfeatures2d

main:
	@$(CC) $(LFLAGS) main.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

Util:
	@$(CC) $(LFLAGS) Util.hpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES)

square_detect:
	@$(CC) $(LFLAGS) square_detect.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

camera_calibration:
	@$(CC) $(LFLAGS) camera_calibration.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

video_stream:
	@$(CC) $(LFLAGS) video_stream.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

cross_matrix_test:
	@$(CC) $(LFLAGS) cross_matrix_test.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

correspondences:
	@$(CC) $(LFLAGS) CameraData.cpp correspondences.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

snap_pictures:
	@$(CC) $(LFLAGS) snap_pictures.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

CameraData:
	@$(CC) $(LFLAGS) CameraData.hpp CameraData.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES)

Matcher:
	@$(CC) $(LFLAGS) Matcher.hpp Matcher.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES)

VanillaTracker:
	@$(CC) $(LFLAGS) CameraData.cpp Util.hpp Tracker.cpp VanillaTracker.hpp VanillaTracker.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES)

run:
	@./main.o

clean:
	rm *.o
	rm *.gch

