import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

class AnalyzerClient {
  final String baseUrl;
  final bool demoMode;

  AnalyzerClient({String? baseUrl})
      : baseUrl = (baseUrl == null || baseUrl.isEmpty)
            ? const String.fromEnvironment(
                'ANALYZER_API_BASE_URL',
                defaultValue: 'http://localhost:8000',
              )
            : baseUrl,
        demoMode = const bool.fromEnvironment('DEMO_MODE', defaultValue: false);

  Future<AnalyzeResult> analyze({
    required Uint8List imageBytes,
    String filename = 'tachograph.jpg',
    String? chartType,
    double? midnightOffsetDeg,
  }) async {
    if (demoMode) {
      final dummy = {
        'totalDrivingMinutes': 390,
        'totalStopMinutes': 80,
        'needsReviewMinutes': 0,
        'segments': [
          {
            'start': '08:00',
            'end': '12:30',
            'type': 'driving',
            'confidence': 'high',
          },
          {
            'start': '12:30',
            'end': '13:50',
            'type': 'stop',
            'confidence': 'mid',
          },
          {
            'start': '13:50',
            'end': '18:00',
            'type': 'driving',
            'confidence': 'high',
          },
        ],
      };
      return AnalyzeResult.fromJson(dummy);
    }

    final uri = Uri.parse('$baseUrl/analyze');

    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: filename),
      );

    if (chartType != null && chartType.isNotEmpty) {
      req.fields['chartType'] = chartType;
    }

    if (midnightOffsetDeg != null) {
      req.fields['midnightOffsetDeg'] = midnightOffsetDeg.toString();
    }

    final res = await req.send();
    final body = await res.stream.bytesToString();

    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: $body');
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map) {
      throw Exception('Invalid response: $body');
    }

    return AnalyzeResult.fromJson(decoded.cast<String, dynamic>());
  }
}
