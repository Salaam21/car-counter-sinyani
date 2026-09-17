import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'detector.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CarCounterApp());
}

class CarCounterApp extends StatelessWidget {
  const CarCounterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Car Counter by Sinyani',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF3DDC97),
          secondary: Color(0xFFFFC857),
          surface: Color(0xFF12161C),
        ),
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
      ),
      home: const CounterHomePage(),
    );
  }
}

class CounterHomePage extends StatefulWidget {
  const CounterHomePage({super.key});

  @override
  State<CounterHomePage> createState() => _CounterHomePageState();
}

class _CounterHomePageState extends State<CounterHomePage> {
  CameraController? _camera;
  VehicleDetector? _detector;
  List<Detection> _detections = const [];
  Map<String, int> _counts = {
    for (final v in VehicleDetector.vehicleClasses) v: 0,
  };
  String? _error;
  bool _busy = false;
  bool _ready = false;
  double _conf = 0.45;
  int _frame = 0;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    try {
      final cam = await Permission.camera.request();
      if (!cam.isGranted) {
        setState(() => _error = 'Camera permission is required.');
        return;
      }
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found on this phone.');
        return;
      }
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      final detector = await VehicleDetector.load();
      if (!mounted) return;
      setState(() {
        _camera = controller;
        _detector = detector;
        _ready = true;
      });
      await controller.startImageStream(_onFrame);
    } catch (e) {
      setState(() => _error = 'Could not start camera/model: $e');
    }
  }

  void _onFrame(CameraImage image) {
    if (_busy || _detector == null) return;
    _frame++;
    if (_frame % 4 != 0) return;
    _busy = true;
    Future(() {
      try {
        final dets = _detector!.detect(image, confGate: _conf);
        final counts = _detector!.countsFrom(dets);
        if (!mounted) return;
        setState(() {
          _detections = dets;
          _counts = counts;
        });
      } catch (_) {
      } finally {
        _busy = false;
      }
    });
  }

  @override
  void dispose() {
    _camera?.dispose();
    _detector?.close();
    super.dispose();
  }

  int get _total => _counts.values.fold(0, (a, b) => a + b);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
              child: Row(
                children: [
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Car Counter',
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.w700,
                            letterSpacing: -0.5,
                          ),
                        ),
                        Text(
                          'by Sinyani',
                          style: TextStyle(
                            color: Color(0xFF3DDC97),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1A222C),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF2A3542)),
                    ),
                    child: Column(
                      children: [
                        Text(
                          '$_total',
                          style: const TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFFFFC857),
                          ),
                        ),
                        const Text(
                          'vehicles',
                          style: TextStyle(fontSize: 11, color: Colors.white70),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: _buildPreview(),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: VehicleDetector.vehicleClasses.map((label) {
                  final n = _counts[label] ?? 0;
                  return _ChipCount(label: label, count: n);
                }).toList(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 4),
              child: Row(
                children: [
                  const Text(
                    'Confidence',
                    style: TextStyle(color: Colors.white70),
                  ),
                  Expanded(
                    child: Slider(
                      value: _conf,
                      min: 0.2,
                      max: 0.8,
                      onChanged: (v) => setState(() => _conf = v),
                    ),
                  ),
                  Text(_conf.toStringAsFixed(2)),
                ],
              ),
            ),
            const Padding(
              padding: EdgeInsets.only(bottom: 12),
              child: Text(
                'Same SSD MobileNet model as Blindspot · on-device only',
                style: TextStyle(fontSize: 11, color: Colors.white38),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreview() {
    if (_error != null) {
      return Container(
        color: const Color(0xFF1A222C),
        alignment: Alignment.center,
        padding: const EdgeInsets.all(24),
        child: Text(_error!, textAlign: TextAlign.center),
      );
    }
    if (!_ready || _camera == null || !_camera!.value.isInitialized) {
      return Container(
        color: const Color(0xFF1A222C),
        alignment: Alignment.center,
        child: const CircularProgressIndicator(color: Color(0xFF3DDC97)),
      );
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_camera!),
        CustomPaint(painter: _BoxPainter(_detections)),
        Positioned(
          left: 12,
          bottom: 12,
          right: 12,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Text(
              _total == 0
                  ? 'Point at the road — counting cars, buses, trucks…'
                  : _summaryText(),
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        ),
      ],
    );
  }

  String _summaryText() {
    final parts = _counts.entries
        .where((e) => e.value > 0)
        .map((e) => '${e.value} ${e.key}${e.value == 1 ? '' : 's'}')
        .join(', ');
    return 'Live count: $_total ($parts)';
  }
}

class _ChipCount extends StatelessWidget {
  const _ChipCount({required this.label, required this.count});

  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: count > 0 ? const Color(0xFF16352B) : const Color(0xFF1A222C),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
          color: count > 0 ? const Color(0xFF3DDC97) : const Color(0xFF2A3542),
        ),
      ),
      child: Text(
        '$label  $count',
        style: TextStyle(
          fontWeight: FontWeight.w600,
          color: count > 0 ? const Color(0xFF3DDC97) : Colors.white70,
        ),
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.detections);

  final List<Detection> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xFF3DDC97);
    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.xmin * size.width,
        d.ymin * size.height,
        d.xmax * size.width,
        d.ymax * size.height,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        paint,
      );
      textPainter.text = TextSpan(
        text: '${d.label} ${(d.score * 100).round()}%',
        style: const TextStyle(
          color: Color(0xFFFFC857),
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left + 4, rect.top - 16));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) =>
      oldDelegate.detections != detections;
}
