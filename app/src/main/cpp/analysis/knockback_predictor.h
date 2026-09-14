#ifndef ESP_KNOCKBACK_PREDICTOR_H
#define ESP_KNOCKBACK_PREDICTOR_H

#include "../detector/bounding_box.h"
#include "../utils/vector2.h"

namespace ESP {

class KnockbackPredictor {
public:
    struct ConfigData {
        float predictionTimeSeconds = 0.18f;
        float gravityPixelsPerSecondSquared = 1450.0f;
        float velocitySmoothing = 0.65f;
        float maxVelocityPixelsPerSecond = 4500.0f;
        float minimumMotionPixels = 1.5f;

        float pushVectorLength = 130.0f;
        float edgeBias = 1.25f;
    };

    struct Prediction {
        bool valid = false;

        Vector2 currentPosition;
        Vector2 velocity;
        Vector2 predictedPosition;

        Vector2 pushVector;
        Vector2 pushEnd;

        float deltaTimeSeconds = 0.0f;
        bool airborneCandidate = false;
    };

    explicit KnockbackPredictor(
        const ConfigData& config = ConfigData());

    Prediction update(
        const BoundingBox& target,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents,
        const Vector2& selfPosition,
        double timestampSeconds);

    void reset();

    const Prediction& getLastPrediction() const {
        return lastPrediction_;
    }

private:
    ConfigData config_;

    bool hasPrevious_;
    Vector2 previousPosition_;
    Vector2 filteredVelocity_;
    double previousTimestamp_;

    Prediction lastPrediction_;

    Vector2 calculatePushVector(
        const Vector2& predictedPosition,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents,
        const Vector2& selfPosition) const;

    bool isOutsideZone(
        const Vector2& position,
        const Vector2& zoneCenter,
        const Vector2& zoneHalfExtents) const;

    Vector2 normalizeSafe(
        const Vector2& value) const;
};

} // namespace ESP

#endif // ESP_KNOCKBACK_PREDICTOR_H
