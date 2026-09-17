# Car Counter by Sinyani

Separate Flutter app for **roadside vehicle counting**.

- **Not** Blindspot — new package id `com.sinyani.car_counter_sinyani`
- Uses the **same SSD MobileNet v1 TFLite** weights Blindspot ships (`assets/ssd_mobilenet_v1.tflite`)
- Counts: bicycle, car, motorcycle, bus, truck

## Run on phone

```bash
flutter pub get
flutter run --release
```

## Shareable APK

```bash
flutter build apk --release
```

APK path: `build/app/outputs/flutter-apk/app-release.apk`

## Author

Munyeme Sinyani — CSC 6302 Computer Vision capstone companion.
