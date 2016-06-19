CC          = c++
LFLAGS      = 
CFLAGS      = -c -Wall -pedantic
#MAIN_OBJS   =  Util.o Camera.o MultiView.o
INCLUDE_DIR = -I/usr/local/include/opencv -I/usr/local/include/opencv2
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

run.o: main.o
	$(CC) $(LFLAGS) $(OBJS) main.o -o run.o $(INCLUDE_DIR) $(LIBRARIES)

main.o:
	$(CC) $(CFLAGS) main.cpp $(INCLUDE_DIR)

clean:
	rm -f *.o
	rm -f *.gch

