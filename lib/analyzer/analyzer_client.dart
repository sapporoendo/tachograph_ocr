import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

class AnalyzerClient {
  final String baseUrl;
  final bool demoMode;

  static String _normalizeBaseUrl(String url) {
    var u = url.trim();
    while (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    return u;
  }

  AnalyzerClient({String? baseUrl, bool? demoMode})
      : baseUrl = _normalizeBaseUrl(
          (baseUrl == null || baseUrl.isEmpty)
              ? const String.fromEnvironment(
                  'ANALYZER_API_BASE_URL',
                  defaultValue: 'http://127.0.0.1:8000',
                )
              : baseUrl,
        ),
        demoMode = demoMode ??
            const bool.fromEnvironment('DEMO_MODE', defaultValue: false);

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

  Future<AnalyzeResult> createRecord({
    required Uint8List imageBytes,
    required String filename,
    required String driverName,
    required String vehicleNo,
    double? distanceKm,
    String? chartType,
    double? midnightOffsetDeg,
  }) async {
    if (demoMode) {
      final dummy = {
        'recordId': 'demo_record_001',
        'totalDrivingMinutes': 390,
        'totalStopMinutes': 80,
        'needsReviewMinutes': 0,
        'segments': [
          {
            'start': '07:20',
            'end': '16:00',
            'type': 'DRIVE',
            'confidence': 'demo',
            'durationMinutes': 520,
          },
        ],
      };
      return AnalyzeResult.fromJson(dummy);
    }

    final uri = Uri.parse('$baseUrl/records');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: filename),
      )
      ..fields['driverName'] = driverName
      ..fields['vehicleNo'] = vehicleNo;

    if (distanceKm != null) {
      req.fields['distanceKm'] = distanceKm.toString();
    }
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
