CC          = g++
CFLAGS      = -Wall -c $(DEBUG) -O3
LFLAGS      = -Wall $(DEBUG)
INCLUDE_DIR = -I/usr/local/include/opencv -I/usr/local/include/opencv2
LIBRARY_DIR = -L/usr/local/lib
LIBRARIES   = -lopencv_calib3d    \
              -lopencv_core       \
              -lopencv_features2d \
              -lopencv_flann      \
              -lopencv_highgui    \
              -lopencv_imgproc    \
              -lopencv_ml         \
              -lopencv_objdetect  \
              -lopencv_photo      \
              -lopencv_stitching  \
              -lopencv_superres   \
              -lopencv_ts         \
              -lopencv_video      \
              -lopencv_videoio    \
              -lopencv_videostab

main:
	@$(CC) $(LFLAGS) main.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

square_detect:
	@$(CC) $(LFLAGS) square_detect.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

camera_calibration:
	@$(CC) $(LFLAGS) camera_calibration.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

video_stream:
	@$(CC) $(LFLAGS) video_stream.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

cross_matrix_test:
	@$(CC) $(LFLAGS) cross_matrix_test.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

correspondences:
	@$(CC) $(LFLAGS) correspondences.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

run:
	@./main.o

clean:
	rm *.o

