# Smash Legends Visual Mode

This fork configures AimBuddy as a visual-only prototype for Lenovo Legion Y700.

## Current behavior
- Screen capture and existing YOLO/NCNN pipeline are unchanged.
- Touch/key injection and aimbot are not required for the Smash Visual Mode.
- The overlay can draw detection boxes, center points, a line between the two nearest detected character candidates, and a configurable range circle.
- The current self/opponent assignment is a temporary heuristic: the detection nearest the screen center is treated as the local-player candidate, and the nearest other detection as the opponent. Replace this with a Smash-specific two-class model (`me`, `enemy`) for reliable results.

## Model
The app already supports importing `.param` + `.bin` NCNN models through its existing model catalog/import flow. Put a trained Smash Legends NCNN model there and select it from the model list.

## Build
Run `./gradlew assembleDebug` on an Android/Gradle build environment. This environment cannot download the Gradle 9.1 distribution, so an APK was not built here.
