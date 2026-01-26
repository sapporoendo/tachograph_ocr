// Copyright (c) 2026 Kumiko Naito. All rights reserved.
class Segment {
  final String start;
  final String end;
  final String type;
  final String confidence;
  final int durationMinutes;

  const Segment({
    required this.start,
    required this.end,
    required this.type,
    required this.confidence,
    required this.durationMinutes,
  });

  factory Segment.fromJson(Map<String, dynamic> json) {
    return Segment(
      start: (json['start'] ?? '').toString(),
      end: (json['end'] ?? '').toString(),
      type: (json['type'] ?? '').toString(),
      confidence: (json['confidence'] ?? '').toString(),
      durationMinutes: (json['durationMinutes'] as num?)?.round() ?? 0,
    );
  }
}

class AnalyzeResult {
  final int totalDrivingMinutes;
  final int totalStopMinutes;
  final int needsReviewMinutes;
  final List<Segment> segments;

  final String? recordId;

  final String? errorCode;
  final String? message;
  final String? hint;
  final String? debugImageBase64;
  final String? debugTargetRegisteredBase64;
  final String? diffCleanBase64;
  final Map<String, dynamic>? processImages;
  final Map<String, dynamic>? meta;

  final String? needleTimeHHMM;
  final double? needleAngleDeg;

  const AnalyzeResult({
    required this.totalDrivingMinutes,
    required this.totalStopMinutes,
    required this.needsReviewMinutes,
    required this.segments,
    this.recordId,
    this.errorCode,
    this.message,
    this.hint,
    this.debugImageBase64,
    this.debugTargetRegisteredBase64,
    this.diffCleanBase64,
    this.processImages,
    this.meta,
    this.needleTimeHHMM,
    this.needleAngleDeg,
  });

  factory AnalyzeResult.fromJson(Map<String, dynamic> json) {
    final rawSegments = json['segments'];
    final segments = rawSegments is List
        ? rawSegments
            .whereType<Map>()
            .map((e) => Segment.fromJson(e.cast<String, dynamic>()))
            .toList()
        : <Segment>[];

    final rawMeta = json['meta'];
    final meta = rawMeta is Map ? rawMeta.cast<String, dynamic>() : null;

    final needleTimeHHMM = (meta?['needleTimeHHMM'] ?? json['needleTimeHHMM'])?.toString();
    final needleAngleDeg = (meta?['needleAngleDeg'] as num?)?.toDouble();

    return AnalyzeResult(
      totalDrivingMinutes: (json['totalDrivingMinutes'] as num?)?.round() ?? 0,
      totalStopMinutes: (json['totalStopMinutes'] as num?)?.round() ?? 0,
      needsReviewMinutes: (json['needsReviewMinutes'] as num?)?.round() ?? 0,
      segments: segments,
      recordId: json['recordId']?.toString(),
      errorCode: json['errorCode']?.toString(),
      message: json['message']?.toString(),
      hint: json['hint']?.toString(),
      debugImageBase64: json['debugImageBase64']?.toString(),
      debugTargetRegisteredBase64:
          json['debugTargetRegisteredBase64']?.toString(),
      diffCleanBase64: json['diffCleanBase64']?.toString(),
      processImages: json['processImages'] is Map
          ? (json['processImages'] as Map).cast<String, dynamic>()
          : null,
      meta: meta,
      needleTimeHHMM: needleTimeHHMM,
      needleAngleDeg: needleAngleDeg,
    );
  }
}

String minutesToHm(int minutes) {
  final h = minutes ~/ 60;
  final m = minutes % 60;
  final mm = m.toString().padLeft(2, '0');
  return '$h:$mm';
}
