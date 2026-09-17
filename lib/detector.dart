import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Detection {
  const Detection({
    required this.label,
    required this.score,
    required this.ymin,
    required this.xmin,
    required this.ymax,
    required this.xmax,
  });

  final String label;
  final double score;
  final double ymin;
  final double xmin;
  final double ymax;
  final double xmax;
}

class VehicleDetector {
  VehicleDetector._(this._interpreter, this._labels);

  final Interpreter _interpreter;
  final List<String> _labels;

  static const vehicleClasses = <String>{
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
  };

  static Future<VehicleDetector> load() async {
    final interpreter = await Interpreter.fromAsset(
      'assets/ssd_mobilenet_v1.tflite',
      options: InterpreterOptions()..threads = 4,
    );
    final labelData = await rootBundle.loadString('assets/labels.txt');
    final labels = labelData
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();
    return VehicleDetector._(interpreter, labels);
  }

  void close() => _interpreter.close();

  Map<String, int> emptyCounts() => {for (final v in vehicleClasses) v: 0};

  Map<String, int> countsFrom(List<Detection> dets) {
    final counts = emptyCounts();
    for (final d in dets) {
      counts[d.label] = (counts[d.label] ?? 0) + 1;
    }
    return counts;
  }

  List<Detection> detect(CameraImage image, {double confGate = 0.45}) {
    final rgb = _cameraImageToRgb(image);
    if (rgb == null) return const [];

    final inputTensor = _interpreter.getInputTensor(0);
    final shape = inputTensor.shape; // [1,300,300,3]
    final prepared = _preprocess(rgb, shape);

    final out0 = _interpreter.getOutputTensor(0);
    final out1 = _interpreter.getOutputTensor(1);
    final out2 = _interpreter.getOutputTensor(2);

    final locations = Float32List(out0.numElements());
    final classes = Float32List(out1.numElements());
    final scores = Float32List(out2.numElements());

    final outputBuffers = <int, Object>{
      0: locations.reshape(out0.shape),
      1: classes.reshape(out1.shape),
      2: scores.reshape(out2.shape),
    };

    if (_interpreter.getOutputTensors().length > 3) {
      final out3 = _interpreter.getOutputTensor(3);
      final countBuf = Float32List(out3.numElements());
      outputBuffers[3] = countBuf.reshape(out3.shape);
    }

    _interpreter.runForMultipleInputs([prepared], outputBuffers);

    // Flatten back
    final loc = _flatten(outputBuffers[0]!);
    final cls = _flatten(outputBuffers[1]!);
    final sco = _flatten(outputBuffers[2]!);

    var n = sco.length;
    if (outputBuffers.containsKey(3)) {
      final c = _flatten(outputBuffers[3]!);
      if (c.isNotEmpty) n = c.first.toInt().clamp(0, sco.length);
    } else {
      n = out0.shape.length >= 2 ? out0.shape[1] : sco.length;
    }

    final out = <Detection>[];
    for (var i = 0; i < n; i++) {
      final score = sco[i];
      if (score < confGate) continue;
      final classId = cls[i].round();
      if (classId < 0 || classId >= _labels.length) continue;
      final label = _labels[classId].toLowerCase();
      if (!vehicleClasses.contains(label)) continue;
      final base = i * 4;
      if (base + 3 >= loc.length) break;
      out.add(
        Detection(
          label: label,
          score: score,
          ymin: loc[base],
          xmin: loc[base + 1],
          ymax: loc[base + 2],
          xmax: loc[base + 3],
        ),
      );
    }
    return out;
  }

  Object _preprocess(img.Image rgb, List<int> shape) {
    final h = shape[1];
    final w = shape[2];
    final resized = img.copyResize(rgb, width: w, height: h);
    final input = _interpreter.getInputTensor(0);

    if (input.type == TensorType.uint8) {
      final bytes = Uint8List(h * w * 3);
      var i = 0;
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final p = resized.getPixel(x, y);
          bytes[i++] = p.r.toInt();
          bytes[i++] = p.g.toInt();
          bytes[i++] = p.b.toInt();
        }
      }
      return bytes.reshape(shape);
    }

    final floats = Float32List(h * w * 3);
    var i = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);
        floats[i++] = p.r / 255.0;
        floats[i++] = p.g / 255.0;
        floats[i++] = p.b / 255.0;
      }
    }
    return floats.reshape(shape);
  }

  Float32List _flatten(Object nested) {
    final out = <double>[];
    void walk(dynamic x) {
      if (x is num) {
        out.add(x.toDouble());
      } else if (x is Float32List) {
        out.addAll(x);
      } else if (x is List) {
        for (final e in x) {
          walk(e);
        }
      }
    }

    walk(nested);
    return Float32List.fromList(out);
  }

  img.Image? _cameraImageToRgb(CameraImage image) {
    try {
      if (image.format.group == ImageFormatGroup.yuv420) {
        return _yuv420ToImage(image);
      }
      if (image.format.group == ImageFormatGroup.bgra8888) {
        return img.Image.fromBytes(
          width: image.width,
          height: image.height,
          bytes: image.planes.first.bytes.buffer,
          order: img.ChannelOrder.bgra,
        );
      }
      if (image.format.group == ImageFormatGroup.nv21) {
        return _nv21ToImage(image);
      }
    } catch (_) {
      return null;
    }
    return null;
  }

  img.Image _yuv420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;
    final out = img.Image(width: width, height: height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yp = yPlane.bytes[y * yPlane.bytesPerRow + x];
        final uvIndex = (y >> 1) * uPlane.bytesPerRow + (x >> 1) * uvPixelStride;
        final up = uPlane.bytes[uvIndex.clamp(0, uPlane.bytes.length - 1)];
        final vp = vPlane.bytes[uvIndex.clamp(0, vPlane.bytes.length - 1)];
        final r = (yp + 1.370705 * (vp - 128)).clamp(0, 255).toInt();
        final g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128))
            .clamp(0, 255)
            .toInt();
        final b = (yp + 1.732446 * (up - 128)).clamp(0, 255).toInt();
        out.setPixelRgb(x, y, r, g, b);
      }
    }
    // Many Android rear cameras need 90° rotation for upright preview math
    return img.copyRotate(out, angle: 90);
  }

  img.Image _nv21ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final bytes = image.planes.first.bytes;
    final out = img.Image(width: width, height: height);
    final frameSize = width * height;
    for (var j = 0, yp = 0; j < height; j++) {
      var uvp = frameSize + (j >> 1) * width;
      var u = 0;
      var v = 0;
      for (var i = 0; i < width; i++, yp++) {
        final y = bytes[yp] & 0xff;
        if ((i & 1) == 0) {
          v = bytes[uvp++] & 0xff;
          u = bytes[uvp++] & 0xff;
        }
        final r = (y + 1.370705 * (v - 128)).clamp(0, 255).toInt();
        final g = (y - 0.337633 * (u - 128) - 0.698001 * (v - 128))
            .clamp(0, 255)
            .toInt();
        final b = (y + 1.732446 * (u - 128)).clamp(0, 255).toInt();
        out.setPixelRgb(i, j, r, g, b);
      }
    }
    return img.copyRotate(out, angle: 90);
  }
}
