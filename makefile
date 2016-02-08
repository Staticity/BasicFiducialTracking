CC          = g++
CFLAGS      = -Wall -c $(DEBUG)
LFLAGS      = -Wall $(DEBUG)
INCLUDE_DIR = -I/usr/local/include/opencv -I/usr/local/include/opencv2
LIBRARY_DIR = -L/usr/local/lib
LIBRARIES   = -lopencv_calib3d    \
              -lopencv_contrib    \
              -lopencv_core       \
              -lopencv_features2d \
              -lopencv_flann      \
              -lopencv_highgui    \
              -lopencv_imgproc    \
              -lopencv_legacy     \
              -lopencv_ml         \
              -lopencv_nonfree    \
              -lopencv_objdetect  \
              -lopencv_photo      \
              -lopencv_stitching  \
              -lopencv_superres   \
              -lopencv_ts         \
              -lopencv_video      \
              -lopencv_videostab

main:
	@$(CC) $(LFLAGS) main.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o main.o

camera_calibration:
	@$(CC) $(LFLAGS) camera_calibration.cpp $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARIES) -o camera_calibration.o

run:
	@./main.o

clean:
	rm *.o

