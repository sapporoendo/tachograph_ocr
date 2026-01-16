import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../analyzer/models.dart';

class ResultPage extends StatelessWidget {
  final AnalyzeResult result;

  const ResultPage({super.key, required this.result});

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

  @override
  Widget build(BuildContext context) {
    final csv = _toCsv();

    return Scaffold(
      appBar: AppBar(title: const Text('解析結果')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '運転 合計 ${minutesToHm(result.totalDrivingMinutes)}',
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Text(
              '停止/休憩 合計 ${minutesToHm(result.totalStopMinutes)}',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            Text('要確認 ${result.needsReviewMinutes}分'),
            const SizedBox(height: 12),
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
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.separated(
                itemCount: result.segments.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (context, i) {
                  final s = result.segments[i];
                  return ListTile(
                    title: Text('${s.start} - ${s.end}  ${s.type}'),
                    trailing: Text(s.confidence),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
