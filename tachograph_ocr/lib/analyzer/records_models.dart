class RecordSummary {
  final String id;
  final String createdAt;
  final String? driverName;
  final String? vehicleNo;
  final double? distanceKm;
  final String? chartType;
  final double? midnightOffsetDeg;
  final bool checked;

  const RecordSummary({
    required this.id,
    required this.createdAt,
    this.driverName,
    this.vehicleNo,
    this.distanceKm,
    this.chartType,
    this.midnightOffsetDeg,
    required this.checked,
  });

  factory RecordSummary.fromJson(Map<String, dynamic> json) {
    return RecordSummary(
      id: (json['id'] ?? '').toString(),
      createdAt: (json['createdAt'] ?? '').toString(),
      driverName: json['driverName']?.toString(),
      vehicleNo: json['vehicleNo']?.toString(),
      distanceKm: (json['distanceKm'] as num?)?.toDouble(),
      chartType: json['chartType']?.toString(),
      midnightOffsetDeg: (json['midnightOffsetDeg'] as num?)?.toDouble(),
      checked: (json['checked'] as bool?) ?? false,
    );
  }
}

class RecordDetail {
  final String id;
  final String createdAt;
  final String? driverName;
  final String? vehicleNo;
  final double? distanceKm;
  final String? chartType;
  final double? midnightOffsetDeg;
  final bool checked;
  final String? note;

  final String? imagePath;
  final String? debugImagePath;
  final Map<String, dynamic> analysis;

  const RecordDetail({
    required this.id,
    required this.createdAt,
    this.driverName,
    this.vehicleNo,
    this.distanceKm,
    this.chartType,
    this.midnightOffsetDeg,
    required this.checked,
    this.note,
    this.imagePath,
    this.debugImagePath,
    required this.analysis,
  });

  factory RecordDetail.fromJson(Map<String, dynamic> json) {
    final analysis = json['analysis'];
    return RecordDetail(
      id: (json['id'] ?? '').toString(),
      createdAt: (json['createdAt'] ?? '').toString(),
      driverName: json['driverName']?.toString(),
      vehicleNo: json['vehicleNo']?.toString(),
      distanceKm: (json['distanceKm'] as num?)?.toDouble(),
      chartType: json['chartType']?.toString(),
      midnightOffsetDeg: (json['midnightOffsetDeg'] as num?)?.toDouble(),
      checked: (json['checked'] as bool?) ?? false,
      note: json['note']?.toString(),
      imagePath: json['imagePath']?.toString(),
      debugImagePath: json['debugImagePath']?.toString(),
      analysis: analysis is Map ? analysis.cast<String, dynamic>() : <String, dynamic>{},
    );
  }
}
