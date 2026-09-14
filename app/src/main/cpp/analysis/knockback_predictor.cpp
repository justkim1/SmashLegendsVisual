#include "knockback_predictor.h"

#include <algorithm>
#include <cmath>

namespace ESP {

namespace {

float clampValue(
    float value,
    float minimum,
    float maximum) {

    return std::max(
        minimum,
        std::min(maximum, value));
}

} // namespace

KnockbackPredictor::KnockbackPredictor()
    : KnockbackPredictor(ConfigData{}) {
}

KnockbackPredictor::KnockbackPredictor(
    const ConfigData& config)
    : config_(config),
      hasPrevious_(false),
      previousPosition_(0.0f, 0.0f),
      filteredVelocity_(0.0f, 0.0f),
      previousTimestamp_(0.0),
      lastPrediction_() {
}

void KnockbackPredictor::reset() {
    hasPrevious_ = false;
    previousPosition_ = Vector2(0.0f, 0.0f);
    filteredVelocity_ = Vector2(0.0f, 0.0f);
    previousTimestamp_ = 0.0;
    lastPrediction_ = Prediction();
}

Vector2 KnockbackPredictor::normalizeSafe(
    const Vector2& value) const {

    const float length =
        std::sqrt(
            value.x * value.x +
            value.y * value.y);

    if (length <= 0.0001f) {
        return Vector2(0.0f, 0.0f);
    }

    return Vector2(
        value.x / length,
        value.y / length);
}

bool KnockbackPredictor::isOutsideZone(
    const Vector2& position,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents) const {

    return
        std::abs(position.x - zoneCenter.x) >
            zoneHalfExtents.x ||
        std::abs(position.y - zoneCenter.y) >
            zoneHalfExtents.y;
}

Vector2 KnockbackPredictor::calculatePushVector(
    const Vector2& predictedPosition,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents,
    const Vector2& selfPosition) const {

    Vector2 direction(
        predictedPosition.x - zoneCenter.x,
        predictedPosition.y - zoneCenter.y);

    Vector2 normalized =
        normalizeSafe(direction);

    // 예측 위치가 중앙과 거의 같은 경우에는
    // 자신의 위치를 기준으로 바깥 방향을 계산한다.
    if (normalized.x == 0.0f &&
        normalized.y == 0.0f) {

        direction = Vector2(
            predictedPosition.x - selfPosition.x,
            predictedPosition.y - selfPosition.y);

        normalized = normalizeSafe(direction);
    }

    // 그래도 방향이 없으면 사용하지 않는다.
    if (normalized.x == 0.0f &&
        normalized.y == 0.0f) {

        return Vector2(0.0f, 0.0f);
    }

    // 이미 영역 밖으로 향하는 경우
    // 바깥쪽 방향을 조금 더 강조한다.
    if (isOutsideZone(
            predictedPosition,
            zoneCenter,
            zoneHalfExtents)) {

        normalized.x *= config_.edgeBias;
        normalized.y *= config_.edgeBias;

        normalized =
            normalizeSafe(normalized);
    }

    return Vector2(
        normalized.x * config_.pushVectorLength,
        normalized.y * config_.pushVectorLength);
}

KnockbackPredictor::Prediction
KnockbackPredictor::update(
    const BoundingBox& target,
    const Vector2& zoneCenter,
    const Vector2& zoneHalfExtents,
    const Vector2& selfPosition,
    double timestampSeconds) {

    Prediction prediction;

    const Vector2 currentPosition =
        target.center();

    float deltaTime = 1.0f / 60.0f;

    if (hasPrevious_) {

        const double rawDelta =
            timestampSeconds -
            previousTimestamp_;

        if (rawDelta >= 0.005 &&
            rawDelta <= 0.20) {

            deltaTime =
                static_cast<float>(rawDelta);
        }
    }

    Vector2 measuredVelocity(
        0.0f,
        0.0f);

    if (hasPrevious_) {

        const Vector2 delta(
            currentPosition.x -
                previousPosition_.x,
            currentPosition.y -
                previousPosition_.y);

        const float movement =
            std::sqrt(
                delta.x * delta.x +
                delta.y * delta.y);

        if (movement >=
            config_.minimumMotionPixels) {

            measuredVelocity =
                Vector2(
                    delta.x / deltaTime,
                    delta.y / deltaTime);
        }
    }

    // 비정상적으로 큰 순간 이동은 제한한다.
    const float measuredSpeed =
        std::sqrt(
            measuredVelocity.x *
                measuredVelocity.x +
            measuredVelocity.y *
                measuredVelocity.y);

    if (measuredSpeed >
        config_.maxVelocityPixelsPerSecond) {

        const float scale =
            config_.maxVelocityPixelsPerSecond /
            measuredSpeed;

        measuredVelocity.x *= scale;
        measuredVelocity.y *= scale;
    }

    if (!hasPrevious_) {

        filteredVelocity_ =
            measuredVelocity;

    } else {

        const float smoothing =
            clampValue(
                config_.velocitySmoothing,
                0.0f,
                1.0f);

        filteredVelocity_.x =
            filteredVelocity_.x * smoothing +
            measuredVelocity.x *
                (1.0f - smoothing);

        filteredVelocity_.y =
            filteredVelocity_.y * smoothing +
            measuredVelocity.y *
                (1.0f - smoothing);
    }

    const float horizontalSpeed =
        std::abs(filteredVelocity_.x);

    const float verticalSpeed =
        std::abs(filteredVelocity_.y);

    // 단순 화면 분석용 휴리스틱.
    // 수직 이동이 충분히 발생하면 공중 상태 후보로 본다.
    const bool airborne =
        verticalSpeed >
        std::max(
            40.0f,
            horizontalSpeed * 0.35f);

    const float predictionTime =
        clampValue(
            config_.predictionTimeSeconds,
            0.02f,
            0.50f);

    Vector2 predictedPosition(
        currentPosition.x +
            filteredVelocity_.x *
                predictionTime,

        currentPosition.y +
            filteredVelocity_.y *
                predictionTime +
            0.5f *
            config_.gravityPixelsPerSecondSquared *
            predictionTime *
            predictionTime);

    Vector2 pushVector =
        calculatePushVector(
            predictedPosition,
            zoneCenter,
            zoneHalfExtents,
            selfPosition);

    Vector2 pushEnd(
        predictedPosition.x +
            pushVector.x,
        predictedPosition.y +
            pushVector.y);

    prediction.valid = true;
    prediction.currentPosition =
        currentPosition;

    prediction.velocity =
        filteredVelocity_;

    prediction.predictedPosition =
        predictedPosition;

    prediction.pushVector =
        pushVector;

    prediction.pushEnd =
        pushEnd;

    prediction.deltaTimeSeconds =
        deltaTime;

    prediction.airborneCandidate =
        airborne;

    previousPosition_ =
        currentPosition;

    previousTimestamp_ =
        timestampSeconds;

    hasPrevious_ = true;

    lastPrediction_ =
        prediction;

    return prediction;
}

} // namespace ESP
