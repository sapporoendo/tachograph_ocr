import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:typed_data';

import '../analyzer/models.dart';

class ResultPage extends StatelessWidget {
  final AnalyzeResult result;
  final double? distanceBeforeKm;
  final double? distanceAfterKm;
  final double? distanceTotalKm;

  const ResultPage({
    super.key,
    required this.result,
    this.distanceBeforeKm,
    this.distanceAfterKm,
    this.distanceTotalKm,
  });

  String _toCsv() {
    final lines = <String>[];
    lines.add('driving_minutes,stop_minutes,needs_review_minutes');
    lines.add(
      '${result.totalDrivingMinutes},${result.totalStopMinutes},${result.needsReviewMinutes}',
    );
    lines.add('');
    lines.add('start,end,type,confidence');
    for (final s in result.segments) {
      lines.add('${s.start},${s.end},${s.type},${s.confidence}');
    }
    return lines.join('\n');
  }

  Uint8List? _tryDecodeBase64Image(String? b64) {
    if (b64 == null || b64.isEmpty) return null;
    try {
      return base64Decode(b64);
    } catch (_) {
      return null;
    }
  }

  String _valueOrDash(String? v) => (v == null || v.isEmpty) ? '--' : v;

  String _kmOrDash(double? v) => (v == null) ? '--' : v.toStringAsFixed(1);

  bool _isDrive(String type) => type.trim().toUpperCase() == 'DRIVE';

  bool _isRest(String type) {
    final t = type.trim().toUpperCase();
    return t == 'IDLE' || t == 'STOP';
  }

  String _typeJa(String type) {
    if (_isDrive(type)) return '走行';
    if (_isRest(type)) return '休憩';
    return '不明';
  }

  String _minutesToJa(int minutes) {
    final m = minutes < 0 ? 0 : minutes;
    final h = m ~/ 60;
    final mm = m % 60;
    return '${h}時間${mm}分';
  }

  ({int drive, int rest, int unknown}) _sumByType(List<Segment> segments) {
    var drive = 0;
    var rest = 0;
    var unknown = 0;
    for (final s in segments) {
      final dur = s.durationMinutes;
      if (_isDrive(s.type)) {
        drive += dur;
      } else if (_isRest(s.type)) {
        rest += dur;
      } else {
        unknown += dur;
      }
    }
    return (drive: drive, rest: rest, unknown: unknown);
  }

  @override
  Widget build(BuildContext context) {
    final csv = _toCsv();
    final t =
        result.needleTimeHHMM ?? result.meta?['needleTimeHHMM']?.toString();
    final recordId = result.recordId;
    final debugBytes = _tryDecodeBase64Image(result.debugImageBase64);
    final circle = result.meta?['circle'];
    final polarLog = result.meta?['polarLog']?.toString();

    final totals = _sumByType(result.segments);

    const driveColor = Color(0xFFFF6F00);
    const driveBg = Color(0xFFFFF3E0);
    const restColor = Color(0xFF0277BD);
    const restBg = Color(0xFFE1F5FE);
    const unknownColor = Color(0xFF555555);
    const unknownBg = Color(0xFFF2F2F2);

    return Scaffold(
      appBar: AppBar(
        title: Image.asset(
          'assets/images/takomiru_logo.png',
          height: 28,
          fit: BoxFit.contain,
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFF333333), width: 2),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '運行の全体像',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          '時刻: ${t ?? "--:--"}',
                          style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          '走行前: ${_kmOrDash(distanceBeforeKm)} km',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          '走行後: ${_kmOrDash(distanceAfterKm)} km',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '合計距離: ${_kmOrDash(distanceTotalKm)} km',
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  if (recordId != null && recordId.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(
                      '保存ID: $recordId',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 12),
            Center(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 14,
                ),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFF333333), width: 2),
                ),
                child: Column(
                  children: [
                    Text(
                      '走行 合計: ${_minutesToJa(totals.drive)}',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 38,
                        fontWeight: FontWeight.w900,
                        color: driveColor,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '休憩 合計: ${_minutesToJa(totals.rest)}',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 38,
                        fontWeight: FontWeight.w900,
                        color: restColor,
                      ),
                    ),
                    if (totals.unknown > 0) ...[
                      const SizedBox(height: 6),
                      Text(
                        '不明 合計: ${totals.unknown}分',
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                          color: unknownColor,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            const SizedBox(height: 10),

            if (result.errorCode != null) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: const Color(0xFFCC0000)),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'errorCode: ${result.errorCode}',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    if (result.message != null) Text(result.message!),
                    if (result.hint != null) Text('hint: ${result.hint!}'),
                    const SizedBox(height: 8),
                    const Text('撮り直しのコツ：'),
                    const Text('・反射が入らないようにする'),
                    const Text('・斜めになりすぎないようにする'),
                    const Text('・チャート全体が欠けないようにする'),
                    const Text('・暗すぎないようにする'),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ],

            if (debugBytes != null) ...[
              ExpansionTile(
                title: const Text('デバッグ画像を見る'),
                children: [
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: ConstrainedBox(
                      constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height * 0.6,
                      ),
                      child: InteractiveViewer(
                        minScale: 1.0,
                        maxScale: 4.0,
                        child: Image.memory(debugBytes, fit: BoxFit.contain),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
            ] else if ((result.debugImageBase64 ?? '').isNotEmpty) ...[
              const Text('デバッグ画像の復元に失敗しました'),
              const SizedBox(height: 8),
            ],

            ExpansionTile(
              title: const Text('デバッグ情報'),
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('circle: ${circle ?? "--"}'),
                      Text(
                        'needleAngleDeg: ${result.needleAngleDeg?.toStringAsFixed(1) ?? "--"}',
                      ),
                      Text('needleTimeHHMM: ${_valueOrDash(t)}'),
                      if (polarLog != null) Text('polarLog: $polarLog'),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            if (result.needsReviewMinutes > 0) ...[
              Text(
                '要確認 ${result.needsReviewMinutes}分',
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 10),
            ],
            Row(
              children: [
                Expanded(
                  child: FilledButton(
                    onPressed: () async {
                      await Clipboard.setData(ClipboardData(text: csv));
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('CSVをコピーしました')),
                        );
                      }
                    },
                    child: const Text('CSVをコピー'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            const Text(
              '区間一覧',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900),
            ),
            const SizedBox(height: 8),
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: result.segments.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, i) {
                final s = result.segments[i];

                final isDrive = _isDrive(s.type);
                final isRest = _isRest(s.type);
                final label = _typeJa(s.type);
                final barColor = isDrive
                    ? driveColor
                    : isRest
                    ? restColor
                    : unknownColor;
                final bgColor = isDrive
                    ? driveBg
                    : isRest
                    ? restBg
                    : unknownBg;
                final textColor = barColor;

                return IntrinsicHeight(
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Container(
                        width: 10,
                        decoration: BoxDecoration(
                          color: barColor,
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            color: bgColor,
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(
                              color: const Color(0xFF333333),
                              width: 1,
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                '${s.start} - ${s.end}',
                                style: TextStyle(
                                  fontSize: 28,
                                  fontWeight: FontWeight.w900,
                                  color: textColor,
                                ),
                              ),
                              const SizedBox(height: 6),
                              Row(
                                children: [
                                  Text(
                                    label,
                                    style: TextStyle(
                                      fontSize: 24,
                                      fontWeight: FontWeight.w900,
                                      color: textColor,
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Text(
                                    '${s.durationMinutes}分',
                                    style: TextStyle(
                                      fontSize: 24,
                                      fontWeight: FontWeight.w900,
                                      color: textColor,
                                    ),
                                  ),
                                ],
                              ),
                              if (s.confidence.isNotEmpty) ...[
                                const SizedBox(height: 6),
                                Text(
                                  s.confidence,
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                    color: Color(0xFF333333),
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
