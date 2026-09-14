#include "dominion_target_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ESP {

namespace {

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float safeAspect(const BoundingBox& box) {
    if (box.height <= 0.0f) {
        return 0.0f;
    }

    return box.width / box.height;
}

float boxArea(const BoundingBox& box) {
    return std::max(0.0f, box.width) *
           std::max(0.0f, box.height);
}

} // namespace

DominionTargetSelector::DominionTargetSelector()
    : DominionTargetSelector(ConfigData{}) {
}

DominionTargetSelector::DominionTargetSelector(
    const ConfigData& config)
    : config_(config) {
}

void DominionTargetSelector::reset() {
    tracks_.clear();
    candidates_.clear();
}

bool DominionTargetSelector::isInsideZone(
    const BoundingBox& box,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents) const {

    const Vector2 center = box.center();

    return std::abs(center.x - zoneCenter.x) <=
               zoneHalfExtents.x &&
           std::abs(center.y - zoneCenter.y) <=
               zoneHalfExtents.y;
}

bool DominionTargetSelector::isNearZone(
    const BoundingBox& box,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents) const {

    const Vector2 center = box.center();

    const float expandedX = zoneHalfExtents.x * 1.35f;
    const float expandedY = zoneHalfExtents.y * 1.35f;

    return std::abs(center.x - zoneCenter.x) <= expandedX &&
           std::abs(center.y - zoneCenter.y) <= expandedY;
}

bool DominionTargetSelector::looksLikeEffect(
    const BoundingBox& box,
    float screenWidth,
    float screenHeight) const {

    if (box.width < config_.minWidthPixels ||
        box.height < config_.minHeightPixels) {
        return true;
    }

    const float areaRatio =
        boxArea(box) /
        std::max(1.0f, screenWidth * screenHeight);

    if (areaRatio > config_.maxAreaRatio) {
        return true;
    }

    if (box.width > screenWidth * config_.maxWidthRatio) {
        return true;
    }

    if (box.height > screenHeight * config_.maxHeightRatio) {
        return true;
    }

    const float aspect = safeAspect(box);

    if (aspect > 5.0f || aspect < 0.18f) {
        return true;
    }

    return false;
}

float DominionTargetSelector::findPreviousAspectRatio(
    const BoundingBox& box) const {

    if (tracks_.empty()) {
        return 0.0f;
    }

    const Vector2 currentCenter = box.center();

    float bestDistance =
        std::numeric_limits<float>::max();

    float bestAspect = 0.0f;

    for (const Track& track : tracks_) {
        const float distance =
            Vector2::Distance(currentCenter, track.center);

        const float matchRadius =
            std::max(
                45.0f,
                std::max(box.width, box.height) * 1.25f);

        if (distance <= matchRadius &&
            distance < bestDistance) {

            bestDistance = distance;
            bestAspect = track.aspectRatio;
        }
    }

    return bestAspect;
}

bool DominionTargetSelector::aspectRatioChangedAbruptly(
    float currentAspect,
    float previousAspect) const {

    if (currentAspect <= 0.0f ||
        previousAspect <= 0.0f) {
        return false;
    }

    const float difference =
        std::abs(currentAspect - previousAspect);

    const float normalized =
        difference /
        std::max(
            0.01f,
            std::max(currentAspect, previousAspect));

    return normalized >=
           config_.aspectChangeThreshold;
}

float DominionTargetSelector::calculateScore(
    const BoundingBox& box,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents,
    float screenWidth,
    float screenHeight,
    bool insideZone,
    bool nearZone,
    bool likelyDowned) const {

    float score = 0.0f;

    if (insideZone) {
        score += config_.insideZoneWeight;
    } else if (nearZone) {
        score += config_.nearZoneWeight;
    } else {
        score += config_.outsideZoneWeight;
    }

    score +=
        clamp01(box.confidence) *
        config_.confidenceWeight;

    const float distance =
        Vector2::Distance(box.center(), zoneCenter);

    const float zoneSize =
        std::max(
            1.0f,
            std::max(
                zoneHalfExtents.x,
                zoneHalfExtents.y));

    const float proximity =
        1.0f -
        clamp01(distance / (zoneSize * 2.5f));

    score += proximity * config_.distanceWeight;

    if (likelyDowned) {
        score -= 2.5f;
    }

    if (screenWidth <= 0.0f ||
        screenHeight <= 0.0f) {
        score -= 1.0f;
    }

    return score;
}

void DominionTargetSelector::updateTracks(
    const std::vector<BoundingBox>& boxes,
    float screenWidth,
    float screenHeight) {

    for (Track& track : tracks_) {
        track.age++;
        track.missedFrames++;
    }

    for (const BoundingBox& box : boxes) {

        const Vector2 center = box.center();
        const float aspect = safeAspect(box);

        int bestIndex = -1;
        float bestDistance =
            std::numeric_limits<float>::max();

        for (size_t i = 0; i < tracks_.size(); ++i) {

            const float distance =
                Vector2::Distance(
                    center,
                    tracks_[i].center);

            const float matchRadius =
                std::max(
                    45.0f,
                    std::max(box.width, box.height) *
                        1.25f);

            if (distance <= matchRadius &&
                distance < bestDistance) {

                bestDistance = distance;
                bestIndex =
                    static_cast<int>(i);
            }
        }

        if (bestIndex >= 0) {

            Track& track =
                tracks_[bestIndex];

            track.center = center;
            track.aspectRatio = aspect;
            track.missedFrames = 0;

        } else {

            Track newTrack;
            newTrack.center = center;
            newTrack.aspectRatio = aspect;
            newTrack.age = 0;
            newTrack.missedFrames = 0;

            tracks_.push_back(newTrack);
        }
    }

    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [&](const Track& track) {
                return track.missedFrames >
                       config_.maxTrackAge;
            }),
        tracks_.end());

    (void)screenWidth;
    (void)screenHeight;
}

DominionTargetSelector::Selection
DominionTargetSelector::select(
    const std::vector<BoundingBox>& boxes,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents,
    float screenWidth,
    float screenHeight) {

    candidates_.clear();

    updateTracks(
        boxes,
        screenWidth,
        screenHeight);

    Selection selection;

    float bestScore =
        -std::numeric_limits<float>::max();

    for (size_t i = 0; i < boxes.size(); ++i) {

        const BoundingBox& box = boxes[i];

        if (box.width <= 0.0f ||
            box.height <= 0.0f) {
            continue;
        }

        const bool insideZone =
            isInsideZone(
                box,
                zoneCenter,
                zoneHalfExtents);

        const bool nearZone =
            isNearZone(
                box,
                zoneCenter,
                zoneHalfExtents);

        const bool likelyEffect =
            looksLikeEffect(
                box,
                screenWidth,
                screenHeight);

        if (likelyEffect) {
            continue;
        }

        const float aspect =
            safeAspect(box);

        const float previousAspect =
            findPreviousAspectRatio(box);

        const bool aspectChanged =
            aspectRatioChangedAbruptly(
                aspect,
                previousAspect);

        const bool likelyDowned =
            aspectChanged ||
            aspect < config_.normalMinAspect ||
            aspect > config_.normalMaxAspect;

        Candidate candidate;

        candidate.box = box;
        candidate.insideZone = insideZone;
        candidate.nearZone = nearZone;
        candidate.likelyEffect = likelyEffect;
        candidate.likelyDowned = likelyDowned;
        candidate.aspectRatio = aspect;
        candidate.previousAspectRatio =
            previousAspect;

        candidate.score =
            calculateScore(
                box,
                zoneCenter,
                zoneHalfExtents,
                screenWidth,
                screenHeight,
                insideZone,
                nearZone,
                likelyDowned);

        candidates_.push_back(candidate);

        if (candidate.score > bestScore) {

            bestScore = candidate.score;

            selection.target = box;
            selection.valid = true;
            selection.sourceIndex =
                static_cast<int>(i);
        }
    }

    return selection;
}

} // namespace ESP
