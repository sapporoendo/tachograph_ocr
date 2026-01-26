// Copyright (c) 2026 Kumiko Naito. All rights reserved.
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'models.dart';

class AnalyzerClient {
  final String baseUrl;
  final bool demoMode;

  static const Duration _requestTimeout = Duration(seconds: 120);

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

    http.StreamedResponse res;
    try {
      res = await req.send().timeout(_requestTimeout);
    } on TimeoutException {
      throw Exception('Request timeout (${_requestTimeout.inSeconds}s): $baseUrl/analyze');
    } catch (e) {
      throw Exception('Failed to reach API: $baseUrl/analyze ($e)');
    }
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

    http.StreamedResponse res;
    try {
      res = await req.send().timeout(_requestTimeout);
    } on TimeoutException {
      throw Exception('Request timeout (${_requestTimeout.inSeconds}s): $baseUrl/records');
    } catch (e) {
      throw Exception('Failed to reach API: $baseUrl/records ($e)');
    }
    final body = await res.stream.bytesToString();
    if (res.statusCode == 404) {
      final fallbackUri = Uri.parse('$baseUrl/analyze');
      final fallbackReq = http.MultipartRequest('POST', fallbackUri)
        ..files.add(
          http.MultipartFile.fromBytes('file', imageBytes, filename: filename),
        );
      if (chartType != null && chartType.isNotEmpty) {
        fallbackReq.fields['chartType'] = chartType;
      }
      if (midnightOffsetDeg != null) {
        fallbackReq.fields['midnightOffsetDeg'] = midnightOffsetDeg.toString();
      }

      try {
        final fallbackRes = await fallbackReq.send().timeout(_requestTimeout);
        final fallbackBody = await fallbackRes.stream.bytesToString();
        if (fallbackRes.statusCode != 200) {
          throw Exception('HTTP ${fallbackRes.statusCode}: $fallbackBody');
        }
        final decoded = jsonDecode(fallbackBody);
        if (decoded is! Map) {
          throw Exception('Invalid response: $fallbackBody');
        }
        return AnalyzeResult.fromJson(decoded.cast<String, dynamic>());
      } on TimeoutException {
        throw Exception('Request timeout (${_requestTimeout.inSeconds}s): $baseUrl/analyze');
      } catch (e) {
        throw Exception(
          'Endpoint not found: $baseUrl/records (HTTP 404). ' 
          'Tried fallback $baseUrl/analyze but failed: $e. Response: $body',
        );
      }
    }
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
