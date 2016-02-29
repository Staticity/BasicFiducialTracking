CC          = c++
LFLAGS      = 
CFLAGS      = -c 
MAIN_OBJS   = Camera.o Features.o MultiView.o
DRAW_OBJS   = Features.o
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


main.o: Util.o Camera.o Features.o MultiView.o
	$(CC) $(LFLAGS) $(MAIN_OBJS) main.cpp -o main.o $(INCLUDE_DIR) $(LIBRARIES)

draw_matches.o: Util.o Features.o
	$(CC) $(LFLAGS) $(DRAW_OBJS) draw_matches.cpp -o draw_matches.o $(INCLUDE_DIR) $(LIBRARIES)

MultiView.o: MultiView.hpp MultiView.cpp
	$(CC) $(CFLAGS) MultiView.hpp MultiView.cpp $(INCLUDE_DIR)

Features.o: Features.hpp Features.cpp
	$(CC) $(CFLAGS) Features.hpp Features.cpp $(INCLUDE_DIR)

Camera.o: Camera.hpp Camera.cpp
	$(CC) $(CFLAGS) Camera.hpp Camera.cpp $(INCLUDE_DIR)

Util.o: Util.hpp
	$(CC) $(CFLAGS) Util.hpp $(INCLUDE_DIR)

clean:
	rm -f *.o
	rm -f *.gch

