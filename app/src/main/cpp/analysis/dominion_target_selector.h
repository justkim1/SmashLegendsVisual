#ifndef ESP_DOMINION_TARGET_SELECTOR_H
#define ESP_DOMINION_TARGET_SELECTOR_H

#include "../detector/bounding_box.h"
#include "../utils/vector2.h"

#include <vector>

namespace ESP {

class DominionTargetSelector {
public:
    struct ConfigData {
        float insideZoneWeight = 3.0f;
        float nearZoneWeight = 1.5f;
        float outsideZoneWeight = 0.45f;
        float distanceWeight = 0.75f;
        float confidenceWeight = 1.25f;

        float normalMinAspect = 0.30f;
        float normalMaxAspect = 2.80f;
        float aspectChangeThreshold = 0.75f;

        float maxAreaRatio = 0.22f;
        float maxWidthRatio = 0.70f;
        float maxHeightRatio = 0.90f;

        float minWidthPixels = 8.0f;
        float minHeightPixels = 8.0f;

        int maxTrackAge = 6;
    };

    struct Candidate {
        BoundingBox box;
        float score = 0.0f;
        bool insideZone = false;
        bool nearZone = false;
        bool likelyDowned = false;
        bool likelyEffect = false;
        float aspectRatio = 0.0f;
        float previousAspectRatio = 0.0f;
    };

    struct Selection {
        BoundingBox target;
        bool valid = false;
        int sourceIndex = -1;
    };

    explicit DominionTargetSelector();
    explicit DominionTargetSelector(const ConfigData& config);

    Selection select(
        const std::vector<BoundingBox>& boxes,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents,
        float screenWidth,
        float screenHeight);

    void reset();

    const std::vector<Candidate>& getCandidates() const {
        return candidates_;
    }

private:
    struct Track {
        Vector2 center;
        float aspectRatio = 0.0f;
        int age = 0;
        int missedFrames = 0;
    };

    ConfigData config_;
    std::vector<Track> tracks_;
    std::vector<Candidate> candidates_;

    bool isInsideZone(
        const BoundingBox& box,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents) const;

    bool isNearZone(
        const BoundingBox& box,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents) const;

    bool looksLikeEffect(
        const BoundingBox& box,
        float screenWidth,
        float screenHeight) const;

    float calculateScore(
        const BoundingBox& box,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents,
        float screenWidth,
        float screenHeight,
        bool insideZone,
        bool nearZone,
        bool likelyDowned) const;

    float findPreviousAspectRatio(
        const BoundingBox& box) const;

    bool aspectRatioChangedAbruptly(
        float currentAspect,
        float previousAspect) const;

    void updateTracks(
        const std::vector<BoundingBox>& boxes,
        float screenWidth,
        float screenHeight);
};

} // namespace ESP

#endif // ESP_DOMINION_TARGET_SELECTOR_H
