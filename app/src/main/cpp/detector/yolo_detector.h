#ifndef ESP_YOLO_DETECTOR_H
#define ESP_YOLO_DETECTOR_H

#include <atomic>
#include <mutex>
#include <vector>
#include <android/asset_manager.h>
#include <android/hardware_buffer.h>

#include "bounding_box.h"
#include "../settings.h"
#include "../utils/logger.h"
#include "../utils/memory_pool.h"

namespace ESP {

struct DetectionResult {
    DetectionArray boxes;
    float inferenceTimeMs;

    DetectionResult() : inferenceTimeMs(0.0f) {}

    void clear() {
        boxes.clear();
        inferenceTimeMs = 0.0f;
    }
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    bool initialize(AAssetManager* assetManager,
                    int screenWidth,
                    int screenHeight,
                    const char* modelParamPath = nullptr,
                    const char* modelBinPath = nullptr);

    void shutdown();

    bool detect(AHardwareBuffer* buffer, DetectionResult& result);
    bool detect(AHardwareBuffer* buffer,
                DetectionResult& result,
                int dynamicCropSize);

    void setScreenSize(int screenWidth, int screenHeight) {
        screenWidth_.store(screenWidth, std::memory_order_relaxed);
        screenHeight_.store(screenHeight, std::memory_order_relaxed);
    }

    void setConfidenceThreshold(float threshold) {
        confidenceThreshold_.store(threshold, std::memory_order_relaxed);
    }

    float getConfidenceThreshold() const {
        return confidenceThreshold_.load(std::memory_order_relaxed);
    }

    bool isInitialized() const {
        return initialized_;
    }

    DetectionResult getResult() const;

private:
    // 실제 캡처 버퍼 크기를 읽는다.
    bool getBufferInfo(AHardwareBuffer* buffer,
                       int& width,
                       int& height,
                       int& stride);

    // 픽셀 기반 후보 탐지.
    void analyzeFrame(const uint8_t* pixels,
                      int width,
                      int height,
                      int stride,
                      int cropSize,
                      DetectionResult& result);

    // 후보 영역을 주변 픽셀과 비교해 점수화.
    float scoreCandidate(const uint8_t* pixels,
                         int width,
                         int height,
                         int stride,
                         int x,
                         int y,
                         int w,
                         int h) const;

    // 중복 후보 제거.
    void applyNMS(DetectionArray& boxes);

    // 좌표를 실제 화면 좌표로 변환.
    BoundingBox mapToScreen(float x,
                            float y,
                            float w,
                            float h,
                            int captureWidth,
                            int captureHeight) const;

    std::atomic<int> screenWidth_;
    std::atomic<int> screenHeight_;
    std::atomic<float> confidenceThreshold_;

    bool initialized_;

    int currentCropX_;
    int currentCropY_;
    int currentCaptureWidth_;
    int currentCaptureHeight_;

    mutable std::mutex resultMutex_;
    DetectionResult latestResult_;

    // 프레임 간 간단한 안정화용 상태.
    std::vector<BoundingBox> previousBoxes_;
};

} // namespace ESP

#endif // ESP_YOLO_DETECTOR_H
