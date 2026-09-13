#include "yolo_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ESP {

namespace {

static inline uint8_t clampByte(int v) {
    return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

static inline float pixelRedScore(uint8_t r, uint8_t g, uint8_t b) {
    const float rf = static_cast<float>(r);
    const float gf = static_cast<float>(g);
    const float bf = static_cast<float>(b);

    if (rf < 100.0f) return 0.0f;

    const float dominance =
        (rf - std::max(gf, bf)) / 255.0f;

    return std::max(0.0f, std::min(1.0f, dominance * 2.5f));
}

static inline bool validBox(float x, float y, float w, float h) {
    return w >= 4.0f &&
           h >= 4.0f &&
           w <= 600.0f &&
           h <= 600.0f &&
           x >= -w &&
           y >= -h;
}

} // namespace

YoloDetector::YoloDetector()
    : screenWidth_(2560)
    , screenHeight_(1600)
    , confidenceThreshold_(Config::DEFAULT_CONFIDENCE_THRESHOLD)
    , initialized_(false)
    , currentCropX_(0)
    , currentCropY_(0)
    , currentCaptureWidth_(0)
    , currentCaptureHeight_(0) {
    LOGD("Model-free visual detector created");
}

YoloDetector::~YoloDetector() {
    shutdown();
}

bool YoloDetector::initialize(AAssetManager* /*assetManager*/,
                              int screenWidth,
                              int screenHeight,
                              const char* /*modelParamPath*/,
                              const char* /*modelBinPath*/) {
    screenWidth_.store(screenWidth > 0 ? screenWidth : 2560,
                       std::memory_order_relaxed);

    screenHeight_.store(screenHeight > 0 ? screenHeight : 1600,
                        std::memory_order_relaxed);

    previousBoxes_.clear();
    latestResult_.clear();

    initialized_ = true;

    LOGD("Model-free detector initialized: screen=%dx%d",
         screenWidth_.load(),
         screenHeight_.load());

    return true;
}

void YoloDetector::shutdown() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(resultMutex_);

    previousBoxes_.clear();
    latestResult_.clear();

    initialized_ = false;

    LOGD("Model-free detector shutdown");
}

bool YoloDetector::getBufferInfo(AHardwareBuffer* buffer,
                                 int& width,
                                 int& height,
                                 int& stride) {
    if (!buffer) {
        return false;
    }

    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(buffer, &desc);

    width = static_cast<int>(desc.width);
    height = static_cast<int>(desc.height);
    stride = static_cast<int>(desc.stride);

    if (width <= 0 || height <= 0) {
        return false;
    }

    if (stride <= 0) {
        stride = width;
    }

    return true;
}

bool YoloDetector::detect(AHardwareBuffer* buffer,
                          DetectionResult& result) {
    return detect(buffer, result, Config::CROP_SIZE);
}

bool YoloDetector::detect(AHardwareBuffer* buffer,
                          DetectionResult& result,
                          int dynamicCropSize) {
    result.clear();

    if (!initialized_ || !buffer) {
        return false;
    }

    const auto start =
        std::chrono::steady_clock::now();

    int width = 0;
    int height = 0;
    int stride = 0;

    if (!getBufferInfo(buffer, width, height, stride)) {
        return false;
    }

    currentCaptureWidth_ = width;
    currentCaptureHeight_ = height;

    const int safeCrop =
        std::max(64,
                   std::min(dynamicCropSize,
                            std::min(width, height)));

    currentCropX_ = std::max(0, (width - safeCrop) / 2);
    currentCropY_ = std::max(0, (height - safeCrop) / 2);

    void* mapped = nullptr;

    const int lockResult =
        AHardwareBuffer_lock(
            buffer,
            AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
            -1,
            nullptr,
            &mapped);

    if (lockResult != 0 || mapped == nullptr) {
        LOGW("AHardwareBuffer_lock failed: %d", lockResult);
        return false;
    }

    analyzeFrame(
        static_cast<const uint8_t*>(mapped),
        width,
        height,
        stride,
        safeCrop,
        result);

    AHardwareBuffer_unlock(buffer, nullptr);

    const auto end =
        std::chrono::steady_clock::now();

    const float elapsed =
        std::chrono::duration<float, std::milli>(
            end - start).count();

    result.inferenceTimeMs = elapsed;

    {
        std::lock_guard<std::mutex> lock(resultMutex_);
        latestResult_ = result;
    }

    return true;
}

void YoloDetector::analyzeFrame(const uint8_t* pixels,
                                int width,
                                int height,
                                int stride,
                                int cropSize,
                                DetectionResult& result) {
    if (!pixels || width <= 0 || height <= 0) {
        return;
    }

    const int cropX = currentCropX_;
    const int cropY = currentCropY_;

    const int cropW =
        std::min(cropSize, width - cropX);

    const int cropH =
        std::min(cropSize, height - cropY);

    if (cropW <= 0 || cropH <= 0) {
        return;
    }

    /*
     * Fast first-stage scan.
     *
     * We deliberately do not assume the capture buffer is
     * the same size as the physical Y700 display.
     *
     * All candidate coordinates remain in capture space until
     * mapToScreen() converts them to the actual screen space.
     */
    const int step = 4;

    struct Candidate {
        int x;
        int y;
        int w;
        int h;
        float score;
    };

    Candidate candidates[Config::MAX_DETECTIONS];
    int candidateCount = 0;

    const int minCell = 8;
    const int maxCell = 160;

    for (int y = cropY; y < cropY + cropH; y += step) {
        for (int x = cropX; x < cropX + cropW; x += step) {

            const size_t offset =
                (static_cast<size_t>(y) *
                 static_cast<size_t>(stride) +
                 static_cast<size_t>(x)) * 4u;

            const uint8_t r = pixels[offset + 0];
            const uint8_t g = pixels[offset + 1];
            const uint8_t b = pixels[offset + 2];

            /*
             * First-stage visual feature:
             * strong red / warm-color contrast.
             *
             * This is intentionally only a candidate generator,
             * not a claim that every red pixel is an enemy.
             */
            const float redScore =
                pixelRedScore(r, g, b);

            if (redScore < 0.42f) {
                continue;
            }

            int left = x;
            int right = x;
            int top = y;
            int bottom = y;

            const int searchRadius = 12;

            for (int yy = std::max(cropY, y - searchRadius);
                 yy <= std::min(cropY + cropH - 1,
                                 y + searchRadius);
                 yy += 4) {

                for (int xx = std::max(cropX, x - searchRadius);
                     xx <= std::min(cropX + cropW - 1,
                                    x + searchRadius);
                     xx += 4) {

                    const size_t p =
                        (static_cast<size_t>(yy) *
                         static_cast<size_t>(stride) +
                         static_cast<size_t>(xx)) * 4u;

                    const float s =
                        pixelRedScore(
                            pixels[p + 0],
                            pixels[p + 1],
                            pixels[p + 2]);

                    if (s >= 0.42f) {
                        left = std::min(left, xx);
                        right = std::max(right, xx);
                        top = std::min(top, yy);
                        bottom = std::max(bottom, yy);
                    }
                }
            }

            const int bw = right - left + 1;
            const int bh = bottom - top + 1;

            if (bw < minCell || bh < minCell ||
                bw > maxCell || bh > maxCell) {
                continue;
            }

            const float shape =
                std::min(
                    static_cast<float>(bw) /
                        static_cast<float>(bh),
                    static_cast<float>(bh) /
                        static_cast<float>(bw));

            const float shapeScore =
                std::max(0.0f,
                         std::min(1.0f, shape));

            const float score =
                redScore * 0.75f +
                shapeScore * 0.25f;

            if (score <
                confidenceThreshold_.load(
                    std::memory_order_relaxed)) {
                continue;
            }

            if (candidateCount <
                Config::MAX_DETECTIONS) {

                candidates[candidateCount++] =
                    {left, top, bw, bh, score};
            }
        }
    }

    /*
     * Convert candidates to the existing BoundingBox format.
     * The rest of the project can therefore continue using
     * Tracker / Renderer without knowing that YOLO was removed.
     */
    for (int i = 0;
         i < candidateCount &&
         result.boxes.size() < Config::MAX_DETECTIONS;
         ++i) {

        const Candidate& c = candidates[i];

        BoundingBox box =
            mapToScreen(
                static_cast<float>(c.x),
                static_cast<float>(c.y),
                static_cast<float>(c.w),
                static_cast<float>(c.h),
                width,
                height);

        box.confidence = c.score;
        box.classId = Config::ENEMY_CLASS_ID;

        if (!validBox(box.x,
                      box.y,
                      box.width,
                      box.height)) {
            continue;
        }

        result.boxes.push(box);
    }

    applyNMS(result.boxes);
}

float YoloDetector::scoreCandidate(const uint8_t* pixels,
                                    int width,
                                    int height,
                                    int stride,
                                    int x,
                                    int y,
                                    int w,
                                    int h) const {
    if (!pixels ||
        x < 0 || y < 0 ||
        x >= width || y >= height ||
        w <= 0 || h <= 0) {
        return 0.0f;
    }

    const int x2 =
        std::min(width - 1, x + w - 1);

    const int y2 =
        std::min(height - 1, y + h - 1);

    float total = 0.0f;
    int count = 0;

    const int step = 4;

    for (int yy = y; yy <= y2; yy += step) {
        for (int xx = x; xx <= x2; xx += step) {
            const size_t p =
                (static_cast<size_t>(yy) *
                 static_cast<size_t>(stride) +
                 static_cast<size_t>(xx)) * 4u;

            total += pixelRedScore(
                pixels[p + 0],
                pixels[p + 1],
                pixels[p + 2]);

            ++count;
        }
    }

    return count > 0 ? total / count : 0.0f;
}

BoundingBox YoloDetector::mapToScreen(float x,
                                      float y,
                                      float w,
                                      float h,
                                      int captureWidth,
                                      int captureHeight) const {
    const float screenW =
        static_cast<float>(
            screenWidth_.load(std::memory_order_relaxed));

    const float screenH =
        static_cast<float>(
            screenHeight_.load(std::memory_order_relaxed));

    const float safeCaptureW =
        std::max(1.0f,
                 static_cast<float>(captureWidth));

    const float safeCaptureH =
        std::max(1.0f,
                 static_cast<float>(captureHeight));

    const float sx =
        screenW / safeCaptureW;

    const float sy =
        screenH / safeCaptureH;

    return BoundingBox(
        x * sx,
        y * sy,
        w * sx,
        h * sy,
        0.0f,
        Config::ENEMY_CLASS_ID);
}

void YoloDetector::applyNMS(DetectionArray& boxes) {
    const size_t count = boxes.size();

    if (count <= 1) {
        return;
    }

    bool suppressed[Config::MAX_DETECTIONS]{};

    DetectionArray output;

    for (size_t i = 0;
         i < count;
         ++i) {

        if (suppressed[i]) {
            continue;
        }

        size_t best = i;

        for (size_t j = i + 1;
             j < count;
             ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (boxes[j].confidence >
                boxes[best].confidence) {
                best = j;
            }
        }

        if (best != i) {
            std::swap(boxes[i], boxes[best]);
        }

        output.push(boxes[i]);

        for (size_t j = i + 1;
             j < count;
             ++j) {

            if (suppressed[j]) {
                continue;
            }

            if (boxes[i].iou(boxes[j]) >
                Config::NMS_IOU_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }

    boxes.clear();

    for (size_t i = 0;
         i < output.size() &&
         boxes.size() < Config::MAX_DETECTIONS;
         ++i) {
        boxes.push(output[i]);
    }
}

DetectionResult YoloDetector::getResult() const {
    std::lock_guard<std::mutex> lock(resultMutex_);
    return latestResult_;
}

} // namespace ESP
