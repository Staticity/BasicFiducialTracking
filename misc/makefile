CC          = c++
LFLAGS      = 
CFLAGS      = -c 
OBJS        =  Camera.o Tracker.o Matcher.o VanillaTracker.o FlowMatcher.o VanillaMatcher.o HybridMatcher.o
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

run_two_view.o: two_view.o
	$(CC) $(LFLAGS) $(OBJS) two_view.o -o run_2_view.o $(INCLUDE_DIR) $(LIBRARIES)

main.o: Camera.o VanillaTracker.o HybridMatcher.o
	$(CC) $(CFLAGS) main.cpp $(INCLUDE_DIR)

two_view.o: Camera.o VanillaTracker.o HybridMatcher.o
	$(CC) $(CFLAGS) two_view.cpp $(INCLUDE_DIR)

correspondences.o: Camera.o correspondences.cpp
	$(CC) $(LFLAGS) correspondences.cpp Camera.o -o correspondences.o $(INCLUDE_DIR) $(LIBRARIES)

square_detect.o: square_detect.cpp
	$(CC) $(LFLAGS) square_detect.cpp -o square_detect.o $(INCLUDE_DIR) $(LIBRARIES)

snap_pictures.o:
	$(CC) $(LFLAGS) snap_pictures.cpp -o snap_pictures.o $(INCLUDE_DIR) $(LIBRARIES)

video_stream.o:
	$(CC) $(LFLAGS) video_stream.cpp -o video_stream.o $(INCLUDE_DIR) $(LIBRARIES)

Util.o:
	$(CC) $(CFLAGS) Util.hpp $(INCLUDE_DIR)

VanillaTracker.o: Util.o Tracker.o VanillaTracker.cpp
	$(CC) $(CFLAGS) VanillaTracker.hpp VanillaTracker.cpp $(INCLUDE_DIR)

Tracker.o: Camera.o Tracker.cpp
	$(CC) $(CFLAGS) Tracker.hpp Tracker.cpp $(INCLUDE_DIR)

FlowMatcher.o: Matcher.o FlowMatcher.cpp
	$(CC) $(CFLAGS) FlowMatcher.hpp FlowMatcher.cpp $(INCLUDE_DIR)

VanillaMatcher.o: Matcher.o VanillaMatcher.cpp
	$(CC) $(CFLAGS) VanillaMatcher.hpp VanillaMatcher.cpp $(INCLUDE_DIR)
    
HybridMatcher.o: VanillaMatcher.o FlowMatcher.o HybridMatcher.cpp
	$(CC) $(CFLAGS) HybridMatcher.hpp HybridMatcher.cpp $(INCLUDE_DIR)
    
Matcher.o: Matcher.cpp
	$(CC) $(CFLAGS) Matcher.hpp Matcher.cpp $(INCLUDE_DIR)

Camera.o: Camera.cpp
	$(CC) $(CFLAGS) Camera.hpp Camera.cpp $(INCLUDE_DIR)

clean:
	rm -f *.o
	rm -f *.gch

