import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tachograph OCR MVP',
      theme: ThemeData(useMaterial3: true),
      home: const OcrDemoPage(),
    );
  }
}

class OcrDemoPage extends StatefulWidget {
  const OcrDemoPage({super.key});
  @override
  State<OcrDemoPage> createState() => _OcrDemoPageState();
}

class _OcrDemoPageState extends State<OcrDemoPage> {
  bool _running = false;
  String _text = '';
  Uint8List? _orig;
  Uint8List? _crop;

  Future<void> _run() async {
    setState(() {
      _running = true;
      _text = '';
    });

    try {
      // 1) assetsから画像を読み込み
      final data = await rootBundle.load('assets/tacho.jpg');
      final origBytes = data.buffer.asUint8List();

      // 2) 中心クロップ（比率は後で調整）
      final cropBytes = cropCenter(origBytes, wRatio: 0.60, hRatio: 0.35);

      // 3) OCRに投げる
      final ocrText = await runOcr(cropBytes);

      setState(() {
        _orig = origBytes;
        _crop = cropBytes;
        _text = ocrText;
      });
    } catch (e) {
      setState(() {
        _text = 'ERROR: $e';
      });
    } finally {
      setState(() {
        _running = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tachograph OCR MVP')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: _running ? null : _run,
              child: Text(_running ? 'OCR中…' : 'assets/tacho.jpg をOCRする'),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Row(
                children: [
                  Expanded(
                    child: _orig == null
                        ? const Center(child: Text('元画像（実行後に表示）'))
                        : Image.memory(_orig!, fit: BoxFit.contain),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _crop == null
                        ? const Center(child: Text('中心クロップ（実行後に表示）'))
                        : Image.memory(_crop!, fit: BoxFit.contain),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SingleChildScrollView(
                  child: Text(_text.isEmpty ? 'OCR結果がここに出ます' : _text),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 中央クロップ（比率）
Uint8List cropCenter(Uint8List bytes, {double wRatio = 0.60, double hRatio = 0.35}) {
  final src = img.decodeImage(bytes);
  if (src == null) return bytes;

  final w = src.width;
  final h = src.height;

  final cropW = (w * wRatio).round();
  final cropH = (h * hRatio).round();

  final x = ((w - cropW) / 2).round();
  final y = ((h - cropH) / 2).round();

  final cropped = img.copyCrop(src, x: x, y: y, width: cropW, height: cropH);
  final jpg = img.encodeJpg(cropped, quality: 90);
  return Uint8List.fromList(jpg);
}

/// OCR.space で全文テキスト取得（MVP用）
Future<String> runOcr(Uint8List imageBytes) async {
  const apiKey = 'K85956699488957'; // ←あなたのキー

  final uri = Uri.parse('https://api.ocr.space/parse/image');
  final req = http.MultipartRequest('POST', uri)
    ..fields['apikey'] = apiKey
    ..fields['language'] = 'jpn'
    ..fields['isOverlayRequired'] = 'false'
    ..fields['OCREngine'] = '2'
    ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: 'crop.jpg'));

  final res = await req.send();
  final body = await res.stream.bytesToString();

  // HTTPが200じゃないなら生の返事を出す
  if (res.statusCode != 200) {
    return 'HTTP ${res.statusCode}\nRAW:\n$body';
  }

  dynamic decoded;
  try {
    decoded = jsonDecode(body);
  } catch (_) {
    // JSONですらない（HTMLとか）場合
    return 'RAW(not JSON):\n$body';
  }

  // 返事がList（配列）だった場合は、まず中身を見せる
  if (decoded is List) {
    return 'RAW(JSON is List):\n$body';
  }

  if (decoded is! Map) {
    return 'RAW(JSON is not Map):\n$body';
  }

  final json = decoded;

  // OCR.space のエラー形式
  if (json['IsErroredOnProcessing'] == true) {
    return 'OCR ERROR: ${json['ErrorMessage']}\n\nRAW:\n$body';
  }

  final results = json['ParsedResults'];
  if (results is! List || results.isEmpty) {
    return 'No ParsedResults\nRAW:\n$body';
  }

  final first = results.first;
  if (first is! Map) {
    return 'ParsedResults[0] not Map\nRAW:\n$body';
  }

  return (first['ParsedText'] ?? '').toString();
}
