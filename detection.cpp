#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include <ctime>
#include <string>
#include <sstream>

class EggDetector {
private:
    // Color ranges for eggs
    cv::Scalar goodEggLower;
    cv::Scalar goodEggUpper;
    cv::Scalar badEggLower;
    cv::Scalar badEggUpper;

    int minArea;
    cv::VideoCapture camera;
    int cameraIndex;

    int resizeWidth;
    int blurKernelSize;
    int morphKernelSize;
    bool showTracks;

    std::map<int, cv::Point2f> trackedEggs;
    int nextEggId;
    int trackBuffer;
    std::map<int, std::deque<cv::Point2f>> trackedPositions;
    int maxTrackHistory;

    std::deque<double> frameTimes;
    double confidenceThreshold;
    double fps;

    struct EggInfo {
        int id;
        std::string type;
        cv::Point2f position;
        cv::Size2f dimensions;
        float angle;
        float area;
        float aspectRatio;
        float confidence;
        float goodConfidence;
        float badConfidence;
    };

public:
    EggDetector() {
        // Initialize color ranges for Good eggs (HSV range)
        goodEggLower = cv::Scalar(0, 0, 220);
        goodEggUpper = cv::Scalar(180, 30, 255);

        // Bad eggs range (HSV range)
        badEggLower = cv::Scalar(20, 20, 200);
        badEggUpper = cv::Scalar(40, 60, 255);

        minArea = 500;
        cameraIndex = 1;

        resizeWidth = 1280;
        blurKernelSize = 5;
        morphKernelSize = 5;
        showTracks = true;

        nextEggId = 1;
        trackBuffer = 10;
        maxTrackHistory = 20;

        confidenceThreshold = 0.5;
        fps = 0;
    }

    bool selectCamera() {
        std::vector<int> availableCameras;

        // Scan first 10 camera indices
        for (int i = 0; i < 10; i++) {
            try {
                cv::VideoCapture cap(i);
                if (cap.isOpened()) {
                    availableCameras.push_back(i);
                    cap.release();
                }
            } catch (const std::exception& e) {
                std::cerr << "Error checking camera " << i << ": " << e.what() << std::endl;
            }
        }

        if (availableCameras.empty()) {
            std::cerr << "No cameras detected!" << std::endl;
            return false;
        }

        std::cout << "Available cameras:" << std::endl;
        for (size_t i = 0; i < availableCameras.size(); i++) {
            std::cout << (i + 1) << ": Camera index " << availableCameras[i] << std::endl;
        }

        // Get user selection
        try {
            int selection;
            std::cout << "Select camera (1-" << availableCameras.size() << "): ";
            std::cin >> selection;

            if (selection >= 1 && selection <= static_cast<int>(availableCameras.size())) {
                cameraIndex = availableCameras[selection - 1];
                return connectCamera(cameraIndex);
            } else {
                std::cerr << "Invalid camera selection!" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Please enter a valid number!" << std::endl;
            return false;
        }
    }

    bool connectCamera(int index = -1) {
        if (index != -1) {
            cameraIndex = index;
        }

        try {
            // Release existing camera if any
            if (camera.isOpened()) {
                camera.release();
            }

            camera.open(cameraIndex);

            if (!camera.isOpened()) {
                std::cerr << "Could not open camera at index " << cameraIndex << std::endl;
                return false;
            }

            // Set camera properties for better quality
            camera.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
            camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

            std::cout << "Connected to camera at index " << cameraIndex << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error connecting to camera: " << e.what() << std::endl;
            return false;
        }
    }

    void trackEggs(std::vector<EggInfo>& eggInfo) {
        std::map<int, cv::Point2f> updatedTracks;
        std::map<int, cv::Point2f> currentPositions;

        for (auto& egg : eggInfo) {
            cv::Point2f pos = egg.position;
            bool foundMatch = false;

            for (const auto& prev : trackedEggs) {
                int prevId = prev.first;
                cv::Point2f prevPos = prev.second;

                if (std::abs(prevPos.x - pos.x) < 50 && std::abs(prevPos.y - pos.y) < 50) {
                    updatedTracks[prevId] = pos;
                    currentPositions[prevId] = pos;
                    egg.id = prevId;
                    foundMatch = true;
                    break;
                }
            }

            // Create new track if no match found
            if (!foundMatch) {
                updatedTracks[nextEggId] = pos;
                currentPositions[nextEggId] = pos;
                egg.id = nextEggId;
                nextEggId++;

                // Initialize track history for new egg
                trackedPositions[egg.id] = std::deque<cv::Point2f>();
            }

            // Update track history
            if (trackedPositions.find(egg.id) != trackedPositions.end()) {
                trackedPositions[egg.id].push_back(pos);
                if (trackedPositions[egg.id].size() > maxTrackHistory) {
                    trackedPositions[egg.id].pop_front();
                }
            } else {
                trackedPositions[egg.id] = std::deque<cv::Point2f>{pos};
            }
        }

        // Update current egg positions
        trackedEggs = updatedTracks;
    }

    bool calibrateColors(const cv::Mat& frame, cv::Rect roiGood = cv::Rect(), cv::Rect roiBad = cv::Rect()) {
        if (frame.empty()) {
            std::cerr << "Cannot calibrate: No frame provided" << std::endl;
            return false;
        }

        if (roiGood.width > 0 && roiGood.height > 0) {
            try {
                // Extract HSV values from good egg sample region
                cv::Mat sample = frame(roiGood);

                // Check sample size
                if (sample.total() < 1000) {
                    std::cerr << "Calibration error: ROI too small. Please select a larger region." << std::endl;
                    return false;
                }

                cv::Mat hsvGood;
                cv::cvtColor(sample, hsvGood, cv::COLOR_BGR2HSV);

                // Split the channels
                std::vector<cv::Mat> hsvChannels;
                cv::split(hsvGood, hsvChannels);

                // Calculate min/max values with some margin
                double hMin, hMax, sMin, sMax, vMin, vMax;

                // For H channel (0)
                cv::minMaxLoc(hsvChannels[0], &hMin, &hMax);
                hMin = std::max(0.0, hMin - 10);
                hMax = std::min(180.0, hMax + 10);

                // For S channel (1)
                cv::minMaxLoc(hsvChannels[1], &sMin, &sMax);
                sMin = std::max(0.0, sMin - 10);
                sMax = std::min(255.0, sMax + 10);

                // For V channel (2)
                cv::minMaxLoc(hsvChannels[2], &vMin, &vMax);
                vMin = std::max(0.0, vMin - 10);
                vMax = std::min(255.0, vMax + 10);

                goodEggLower = cv::Scalar(hMin, sMin, vMin);
                goodEggUpper = cv::Scalar(hMax, sMax, vMax);

                std::cout << "Good egg HSV range calibrated: ["
                         << goodEggLower[0] << "," << goodEggLower[1] << "," << goodEggLower[2]
                         << "] to ["
                         << goodEggUpper[0] << "," << goodEggUpper[1] << "," << goodEggUpper[2]
                         << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error calibrating good egg color: " << e.what() << std::endl;
                return false;
            }
        }

        if (roiBad.width > 0 && roiBad.height > 0) {
            try {
                // Extract HSV values from bad egg sample region
                cv::Mat sample = frame(roiBad);

                // Check sample size
                if (sample.total() < 1000) {
                    std::cerr << "Calibration error: ROI too small. Please select a larger region." << std::endl;
                    return false;
                }

                cv::Mat hsvBad;
                cv::cvtColor(sample, hsvBad, cv::COLOR_BGR2HSV);

                // Split the channels
                std::vector<cv::Mat> hsvChannels;
                cv::split(hsvBad, hsvChannels);

                // Calculate min/max values with some margin
                double hMin, hMax, sMin, sMax, vMin, vMax;

                // For H channel (0)
                cv::minMaxLoc(hsvChannels[0], &hMin, &hMax);
                hMin = std::max(0.0, hMin - 10);
                hMax = std::min(180.0, hMax + 10);

                // For S channel (1)
                cv::minMaxLoc(hsvChannels[1], &sMin, &sMax);
                sMin = std::max(0.0, sMin - 10);
                sMax = std::min(255.0, sMax + 10);

                // For V channel (2)
                cv::minMaxLoc(hsvChannels[2], &vMin, &vMax);
                vMin = std::max(0.0, vMin - 10);
                vMax = std::min(255.0, vMax + 10);

                badEggLower = cv::Scalar(hMin, sMin, vMin);
                badEggUpper = cv::Scalar(hMax, sMax, vMax);

                std::cout << "Bad egg HSV range calibrated: ["
                         << badEggLower[0] << "," << badEggLower[1] << "," << badEggLower[2]
                         << "] to ["
                         << badEggUpper[0] << "," << badEggUpper[1] << "," << badEggUpper[2]
                         << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error calibrating bad egg color: " << e.what() << std::endl;
                return false;
            }
        }

        return true;
    }

    cv::Mat optimizeFrame(const cv::Mat& frame) {
        if (frame.empty()) {
            return cv::Mat();
        }

        try {
            int height = frame.rows;
            int width = frame.cols;

            if (width > resizeWidth) {
                double scale = static_cast<double>(resizeWidth) / width;
                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_AREA);
                return resized;
            }

            return frame;
        } catch (const std::exception& e) {
            std::cerr << "Error optimizing frame: " << e.what() << std::endl;
            return frame;  // Return original frame on error
        }
    }

    std::pair<cv::Mat, std::vector<EggInfo>> processFrame(const cv::Mat& frame) {
        auto startTime = std::chrono::high_resolution_clock::now();
        cv::Mat resultFrame;
        std::vector<EggInfo> eggInfo;

        try {
            if (frame.empty()) {
                return {cv::Mat(), eggInfo};
            }

            cv::Mat optimizedFrame = optimizeFrame(frame);
            if (optimizedFrame.empty()) {
                return {cv::Mat(), eggInfo};
            }

            cv::Mat hsv;
            cv::cvtColor(optimizedFrame, hsv, cv::COLOR_BGR2HSV);

            // Apply color thresholding for both good and bad eggs
            cv::Mat maskGood, maskBad, mask;
            cv::inRange(hsv, goodEggLower, goodEggUpper, maskGood);
            cv::inRange(hsv, badEggLower, badEggUpper, maskBad);
            cv::bitwise_or(maskGood, maskBad, mask);

            // Reduce noise - Enhanced
            cv::GaussianBlur(mask, mask, cv::Size(blurKernelSize, blurKernelSize), 0);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morphKernelSize, morphKernelSize));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            optimizedFrame.copyTo(resultFrame);
            int eggCount = 0;
            int goodCount = 0;
            int badCount = 0;

            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);

                if (area > minArea) {
                    if (contour.size() >= 5) {
                        cv::RotatedRect ellipse = cv::fitEllipse(contour);

                        // Calculate dimensions and aspect ratio
                        cv::Point2f center = ellipse.center;
                        cv::Size2f axes = ellipse.size * 0.5f;  // Half-width and half-height
                        float angle = ellipse.angle;
                        float majorAxis = std::max(axes.width, axes.height);
                        float minorAxis = std::min(axes.width, axes.height);
                        float aspectRatio = minorAxis / majorAxis;

                        cv::Rect boundRect = cv::boundingRect(contour);
                        cv::Mat maskSegment = cv::Mat::zeros(mask.size(), CV_8UC1);
                        cv::drawContours(maskSegment, std::vector<std::vector<cv::Point>>{contour}, -1, 255, -1);

                        cv::Mat goodMatch, badMatch;
                        cv::bitwise_and(maskSegment, maskGood, goodMatch);
                        cv::bitwise_and(maskSegment, maskBad, badMatch);

                        int goodMatchCount = cv::countNonZero(goodMatch);
                        int badMatchCount = cv::countNonZero(badMatch);
                        int totalMatch = std::max(goodMatchCount + badMatchCount, 1);

                        float goodConfidence = static_cast<float>(goodMatchCount) / totalMatch;
                        float badConfidence = static_cast<float>(badMatchCount) / totalMatch;

                        std::string eggType;
                        float confidence;
                        cv::Scalar color;

                        if (goodConfidence >= badConfidence && goodConfidence >= confidenceThreshold) {
                            eggType = "Good Egg";
                            confidence = goodConfidence;
                            goodCount++;
                            color = cv::Scalar(0, 255, 0);
                        } else if (badConfidence > goodConfidence && badConfidence >= confidenceThreshold) {
                            eggType = "Bad Egg";
                            confidence = badConfidence;
                            badCount++;
                            color = cv::Scalar(0, 0, 255);
                        } else {
                            eggType = "Uncertain";
                            confidence = std::max(goodConfidence, badConfidence);
                            color = cv::Scalar(0, 255, 255);
                        }

                        cv::ellipse(resultFrame, ellipse, color, 2);

                        EggInfo egg;
                        egg.id = 0;  // Will be assigned in trackEggs
                        egg.type = eggType;
                        egg.position = center;
                        egg.dimensions = axes * 2.0f;  // Converting back to full width and height
                        egg.angle = angle;
                        egg.area = area;
                        egg.aspectRatio = aspectRatio;
                        egg.confidence = confidence;
                        egg.goodConfidence = goodConfidence;
                        egg.badConfidence = badConfidence;

                        eggInfo.push_back(egg);
                        eggCount++;
                    }
                }
            }

            trackEggs(eggInfo);

            resultFrame = visualizeResults(resultFrame, eggInfo, goodCount, badCount);

            auto endTime = std::chrono::high_resolution_clock::now();
            double processTime = std::chrono::duration<double>(endTime - startTime).count();

            frameTimes.push_back(processTime);
            if (frameTimes.size() > 30) {
                frameTimes.pop_front();
            }

            // Calculate FPS - average of last 30 frame processing times
            if (!frameTimes.empty()) {
                double totalTime = 0;
                for (double time : frameTimes) {
                    totalTime += time;
                }
                fps = 1.0 / (totalTime / frameTimes.size());
            }

            return {resultFrame, eggInfo};
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
            if (!frame.empty()) {
                return {frame, eggInfo};  // Return original frame on error
            }
            return {cv::Mat(), eggInfo};
        }
    }

    cv::Mat visualizeResults(const cv::Mat& frame, const std::vector<EggInfo>& eggInfo, int goodCount, int badCount) {
        try {
            cv::Mat result = frame.clone();
            int totalEggs = eggInfo.empty() ? 1 : eggInfo.size();

            // Draw egg ellipses and labels
            for (const auto& egg : eggInfo) {
                cv::Point2f center = egg.position;
                cv::Size2f axes = egg.dimensions * 0.5f;  // Half width and height for ellipse drawing
                float angle = egg.angle;
                int eggId = egg.id;

                // Set color based on egg type
                cv::Scalar color;
                if (egg.type == "Good Egg") {
                    color = cv::Scalar(0, 255, 0);
                } else if (egg.type == "Bad Egg") {
                    color = cv::Scalar(0, 0, 255);
                } else {
                    color = cv::Scalar(0, 255, 255);  // Yellow
                }

                // Draw ellipse
                cv::ellipse(result, cv::RotatedRect(center, axes * 2.0f, angle), color, 2);

                // Display detailed information
                std::stringstream infoText;
                infoText << "#" << eggId << " " << egg.type << " Conf: " << std::fixed << std::setprecision(2) << egg.confidence;

                cv::putText(result, infoText.str(),
                        cv::Point(static_cast<int>(center.x - axes.width / 2), static_cast<int>(center.y - axes.height / 2) - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

                if (showTracks && trackedPositions.find(eggId) != trackedPositions.end()) {
                    const auto& points = trackedPositions[eggId];
                    if (points.size() > 1) {
                        std::vector<cv::Point> trackPoints;
                        for (const auto& point : points) {
                            trackPoints.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
                        }
                        // Use polylines for smoother trajectory visualization
                        cv::polylines(result, std::vector<std::vector<cv::Point>>{trackPoints}, false, color, 2);
                    }
                }
            }

            // Add statistics overlay
            cv::Mat overlay;
            result.copyTo(overlay);
            int statsHeight = 150;
            cv::rectangle(overlay, cv::Rect(0, 0, 300, statsHeight), cv::Scalar(0, 0, 0), -1);

            double alpha = 0.7;
            cv::addWeighted(overlay, alpha, result, 1.0 - alpha, 0, result);

            cv::putText(result, "Total Eggs: " + std::to_string(eggInfo.size()),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            std::stringstream goodText;
            goodText << "Good: " << goodCount << " (" << std::fixed << std::setprecision(1)
                    << (static_cast<float>(goodCount) / totalEggs * 100.0f) << "%)";
            cv::putText(result, goodText.str(),
                    cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            std::stringstream badText;
            badText << "Bad: " << badCount << " (" << std::fixed << std::setprecision(1)
                   << (static_cast<float>(badCount) / totalEggs * 100.0f) << "%)";
            cv::putText(result, badText.str(),
                    cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            std::stringstream fpsText;
            fpsText << "FPS: " << std::fixed << std::setprecision(1) << fps;
            cv::putText(result, fpsText.str(),
                    cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            return result;
        } catch (const std::exception& e) {
            std::cerr << "Error in visualization: " << e.what() << std::endl;
            return frame;  // Return original frame on error
        }
    }

    void runLiveDetection() {
        if (!camera.isOpened()) {
            if (!selectCamera()) {
                std::cout << "Failed to connect to camera" << std::endl;
                return;
            }
        }

        bool calibrationMode = false;
        std::vector<cv::Point> roiPoints;
        std::string calibrationType;

        cv::namedWindow("Egg Detection", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Egg Detection", [](int event, int x, int y, int flags, void* userdata) {
            // This is a placeholder for the mouse callback
            // It will be properly set in the calibration section
        }, this);

        try {
            while (true) {
                cv::Mat frame;
                bool ret = camera.read(frame);
                if (!ret || frame.empty()) {
                    std::cout << "Error: Could not read frame from camera" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }

                // Normal processing mode
                auto [resultFrame, _] = processFrame(frame);
                if (!resultFrame.empty()) {
                    cv::imshow("Egg Detection", resultFrame);
                }

                int key = cv::waitKey(1);

                if (key == 'q') {
                    break;
                } else if (key == 'c') {
                    // Toggle calibration mode
                    calibrationMode = !calibrationMode;
                    roiPoints.clear();

                    if (calibrationMode) {
                        // Ask for calibration type
                        std::cout << "\nEnter calibration type:" << std::endl;
                        std::cout << "1: Good Egg" << std::endl;
                        std::cout << "2: Bad Egg" << std::endl;
                        std::string choice;
                        std::cout << "Select (1/2): ";
                        std::cin >> choice;

                        calibrationType = (choice == "1") ? "Good Egg" : "Bad Egg";
                        std::cout << "Click to select 4 corners around a " << calibrationType << std::endl;

                        // Define mouse callback function for ROI selection
                        cv::setMouseCallback("Egg Detection", [](int event, int x, int y, int flags, void* userdata) {
                            if (event == cv::EVENT_LBUTTONDOWN) {
                                auto* self = static_cast<EggDetector*>(userdata);
                                auto& points = self->roiPoints;
                                if (points.size() < 4) {
                                    points.push_back(cv::Point(x, y));
                                }
                            }
                        }, this);
                    } else {
                        cv::setMouseCallback("Egg Detection", [](int event, int x, int y, int flags, void* userdata) {
                            // Empty callback when not in calibration mode
                        }, this);
                    }
                } else if (key == 13 && calibrationMode && roiPoints.size() == 4) {  // Enter key
                    // Process calibration
                    int xMin = std::min({roiPoints[0].x, roiPoints[1].x, roiPoints[2].x, roiPoints[3].x});
                    int yMin = std::min({roiPoints[0].y, roiPoints[1].y, roiPoints[2].y, roiPoints[3].y});
                    int xMax = std::max({roiPoints[0].x, roiPoints[1].x, roiPoints[2].x, roiPoints[3].x});
                    int yMax = std::max({roiPoints[0].y, roiPoints[1].y, roiPoints[2].y, roiPoints[3].y});

                    cv::Rect roi(xMin, yMin, xMax - xMin, yMax - yMin);

                    if (calibrationType == "Good Egg") {
                        calibrateColors(frame, roi, cv::Rect());
                    } else {
                        calibrateColors(frame, cv::Rect(), roi);
                    }

                    calibrationMode = false;
                    cv::setMouseCallback("Egg Detection", [](int event, int x, int y, int flags, void* userdata) {
                        // Empty callback when not in calibration mode
                    }, this);
                } else if (key == 't') {
                    // Toggle tracking visualization
                    showTracks = !showTracks;
                    std::cout << "Tracking visualization: " << (showTracks ? "ON" : "OFF") << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in detection loop: " << e.what() << std::endl;
        }

        camera.release();
        cv::destroyAllWindows();
        std::cout << "Detection system shutdown" << std::endl;
    }
};

// Global variable for mouse callback
std::vector<cv::Point> roiPoints;

int main() {
    try {
        EggDetector detector;
        detector.runLiveDetection();
    } catch (const std::exception& e) {
        std::cerr << "Critical error in main: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}