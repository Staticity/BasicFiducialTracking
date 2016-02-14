CC          = c++
LFLAGS      = 
CFLAGS      = -c 
OBJS        =  CameraData.o Tracker.o Matcher.o VanillaTracker.o FlowMatcher.o main.o
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
	$(CC) $(LFLAGS) $(OBJS) -o run.o $(INCLUDE_DIR) $(LIBRARIES)

main.o: CameraData.o VanillaTracker.o FlowMatcher.o
	$(CC) $(CFLAGS) main.cpp $(INCLUDE_DIR)

correspondences.o: CameraData.o misc/correspondences.cpp
	$(CC) $(LFLAGS) misc/correspondences.cpp CameraData.o -o correspondences.o $(INCLUDE_DIR) $(LIBRARIES)

square_detect.o: misc/square_detect.cpp
	$(CC) $(LFLAGS) misc/square_detect.cpp -o square_detect.o $(INCLUDE_DIR) $(LIBRARIES)

Util.o:
	$(CC) $(CFLAGS) Util.hpp $(INCLUDE_DIR)

VanillaTracker.o: Util.o Tracker.o VanillaTracker.cpp
	$(CC) $(CFLAGS) VanillaTracker.hpp VanillaTracker.cpp $(INCLUDE_DIR)

Tracker.o: CameraData.o Tracker.cpp
	$(CC) $(CFLAGS) Tracker.hpp Tracker.cpp $(INCLUDE_DIR)

FlowMatcher.o: Matcher.o FlowMatcher.cpp
	$(CC) $(CFLAGS) FlowMatcher.hpp FlowMatcher.cpp $(INCLUDE_DIR)

VanillaMatcher.o: Matcher.o VanillaMatcher.cpp
	$(CC) $(CFLAGS) VanillaMatcher.hpp VanillaMatcher.cpp $(INCLUDE_DIR)
    
Matcher.o: Matcher.cpp
	$(CC) $(CFLAGS) Matcher.hpp Matcher.cpp $(INCLUDE_DIR)

CameraData.o: CameraData.cpp
	$(CC) $(CFLAGS) CameraData.hpp CameraData.cpp $(INCLUDE_DIR)

clean:
	rm -f *.o
	rm -f *.gch

