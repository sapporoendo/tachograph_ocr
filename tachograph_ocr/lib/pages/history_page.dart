// Copyright (c) 2026 Kumiko Naito. All rights reserved.
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../analyzer/analyzer_client.dart';
import '../analyzer/records_models.dart';

class HistoryPage extends StatefulWidget {
  final AnalyzerClient client;

  const HistoryPage({super.key, required this.client});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  late Future<List<RecordSummary>> _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<List<RecordSummary>> _load() async {
    final uri = Uri.parse('${widget.client.baseUrl}/records?limit=200');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    final decoded = jsonDecode(res.body);
    if (decoded is! Map) {
      throw Exception('Invalid response: ${res.body}');
    }
    final records = decoded['records'];
    if (records is! List) return const [];
    return records
        .whereType<Map>()
        .map((e) => RecordSummary.fromJson(e.cast<String, dynamic>()))
        .toList();
  }

  Future<void> _reload() async {
    setState(() {
      _future = _load();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Image.asset(
          'assets/images/takomiru_logo.png',
          height: 28,
          fit: BoxFit.contain,
        ),
        actions: [
          IconButton(onPressed: _reload, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: FutureBuilder<List<RecordSummary>>(
        future: _future,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Text('読み込みエラー: ${snap.error}'),
            );
          }
          final rows = snap.data ?? const [];
          if (rows.isEmpty) {
            return const Center(child: Text('履歴がありません'));
          }
          return RefreshIndicator(
            onRefresh: _reload,
            child: ListView.separated(
              itemCount: rows.length,
              separatorBuilder: (context, index) => const Divider(height: 1),
              itemBuilder: (context, i) {
                final r = rows[i];
                final title =
                    '${r.vehicleNo ?? "--"} / ${r.driverName ?? "--"}';
                final subtitle =
                    '${r.createdAt}${r.distanceKm == null ? '' : ' / ${r.distanceKm!.toStringAsFixed(1)} km'}';
                return ListTile(
                  title: Text(
                    title,
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  subtitle: Text(
                    subtitle,
                    style: const TextStyle(fontSize: 14),
                  ),
                  trailing: Icon(
                    r.checked ? Icons.verified : Icons.hourglass_bottom,
                    color: r.checked
                        ? const Color(0xFF2E7D32)
                        : const Color(0xFF666666),
                    size: 28,
                  ),
                  minVerticalPadding: 16,
                  onTap: () async {
                    await Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => RecordDetailPage(
                          client: widget.client,
                          recordId: r.id,
                        ),
                      ),
                    );
                    await _reload();
                  },
                );
              },
            ),
          );
        },
      ),
    );
  }
}

class RecordDetailPage extends StatefulWidget {
  final AnalyzerClient client;
  final String recordId;

  const RecordDetailPage({
    super.key,
    required this.client,
    required this.recordId,
  });

  @override
  State<RecordDetailPage> createState() => _RecordDetailPageState();
}

class _RecordDetailPageState extends State<RecordDetailPage> {
  late Future<RecordDetail> _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<RecordDetail> _load() async {
    final uri = Uri.parse(
      '${widget.client.baseUrl}/records/${widget.recordId}',
    );
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    final decoded = jsonDecode(res.body);
    if (decoded is! Map) {
      throw Exception('Invalid response: ${res.body}');
    }
    return RecordDetail.fromJson(decoded.cast<String, dynamic>());
  }

  Future<void> _setChecked(bool checked) async {
    final uri = Uri.parse(
      '${widget.client.baseUrl}/records/${widget.recordId}/checked',
    );
    final res = await http.patch(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'checked': checked}),
    );
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    setState(() {
      _future = _load();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Image.asset(
          'assets/images/takomiru_logo.png',
          height: 28,
          fit: BoxFit.contain,
        ),
      ),
      body: FutureBuilder<RecordDetail>(
        future: _future,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Text('読み込みエラー: ${snap.error}'),
            );
          }
          final r = snap.data!;
          final debugUrl =
              '${widget.client.baseUrl}/records/${widget.recordId}/debug.png';
          final title = '${r.vehicleNo ?? "--"} / ${r.driverName ?? "--"}';
          final subtitle =
              '${r.createdAt}${r.distanceKm == null ? '' : ' / ${r.distanceKm!.toStringAsFixed(1)} km'}';

          final segments = r.analysis['segments'];
          final segList = segments is List
              ? segments.whereType<Map>().toList()
              : const <Map>[];

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 4),
              Text(subtitle),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.tonalIcon(
                      onPressed: () => _setChecked(!r.checked),
                      icon: Icon(
                        r.checked
                            ? Icons.check_circle
                            : Icons.radio_button_unchecked,
                      ),
                      label: Text(
                        r.checked ? '確認済み' : '未確認',
                        style: const TextStyle(fontSize: 18),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.network(
                  debugUrl,
                  fit: BoxFit.contain,
                  errorBuilder: (c, e, s) => Text('画像取得エラー: $e'),
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'セグメント',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              if (segList.isEmpty) const Text('セグメントなし'),
              for (final s in segList)
                Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    border: Border.all(color: const Color(0xFFDDDDDD)),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          '${(s['start'] ?? '--')} 〜 ${(s['end'] ?? '--')}',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        (s['type'] ?? 'UNKNOWN').toString(),
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
