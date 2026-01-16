class Segment {
  final String start;
  final String end;
  final String type;
  final String confidence;

  const Segment({
    required this.start,
    required this.end,
    required this.type,
    required this.confidence,
  });

  factory Segment.fromJson(Map<String, dynamic> json) {
    return Segment(
      start: (json['start'] ?? '').toString(),
      end: (json['end'] ?? '').toString(),
      type: (json['type'] ?? '').toString(),
      confidence: (json['confidence'] ?? '').toString(),
    );
  }
}

class AnalyzeResult {
  final int totalDrivingMinutes;
  final int totalStopMinutes;
  final int needsReviewMinutes;
  final List<Segment> segments;

  const AnalyzeResult({
    required this.totalDrivingMinutes,
    required this.totalStopMinutes,
    required this.needsReviewMinutes,
    required this.segments,
  });

  factory AnalyzeResult.fromJson(Map<String, dynamic> json) {
    final rawSegments = json['segments'];
    final segments = rawSegments is List
        ? rawSegments
            .whereType<Map>()
            .map((e) => Segment.fromJson(e.cast<String, dynamic>()))
            .toList()
        : <Segment>[];

    return AnalyzeResult(
      totalDrivingMinutes: (json['totalDrivingMinutes'] as num?)?.round() ?? 0,
      totalStopMinutes: (json['totalStopMinutes'] as num?)?.round() ?? 0,
      needsReviewMinutes: (json['needsReviewMinutes'] as num?)?.round() ?? 0,
      segments: segments,
    );
  }
}

String minutesToHm(int minutes) {
  final h = minutes ~/ 60;
  final m = minutes % 60;
  final mm = m.toString().padLeft(2, '0');
  return '$h:$mm';
}
