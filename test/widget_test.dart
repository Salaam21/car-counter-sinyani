import 'package:flutter_test/flutter_test.dart';

import 'package:car_counter_sinyani/main.dart';

void main() {
  testWidgets('Car Counter app loads', (tester) async {
    await tester.pumpWidget(const CarCounterApp());
    expect(find.textContaining('Car Counter'), findsOneWidget);
    expect(find.textContaining('Sinyani'), findsOneWidget);
  });
}
