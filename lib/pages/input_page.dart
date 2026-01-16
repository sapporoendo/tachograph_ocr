import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../analyzer/analyzer_client.dart';
import '../analyzer/models.dart';
import 'error_page.dart';
import 'result_page.dart';

class InputPage extends StatefulWidget {
  const InputPage({super.key});

  @override
  State<InputPage> createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  final _client = AnalyzerClient();

  bool _running = false;
  Uint8List? _imageBytes;
  String _filename = 'tachograph.jpg';
  String _chartType = '24h';
  double _midnightOffsetDeg = 0;

  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();

    setState(() {
      _imageBytes = bytes;
      _filename = file.name;
    });
  }

  Future<void> _takePhoto() async {
    final picker = ImagePicker();
    final file = await picker.pickImage(source: ImageSource.camera);
    if (file == null) return;
    final bytes = await file.readAsBytes();

    setState(() {
      _imageBytes = bytes;
      _filename = file.name;
    });
  }

  Future<void> _analyze() async {
    final bytes = _imageBytes;
    if (bytes == null) return;

    setState(() {
      _running = true;
    });

    try {
      final AnalyzeResult result = await _client.analyze(
        imageBytes: bytes,
        filename: _filename,
        chartType: _chartType,
        midnightOffsetDeg: _midnightOffsetDeg,
      );

      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ResultPage(result: result)),
      );
    } catch (e) {
      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ErrorPage(message: e.toString())),
      );
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final canRun = !_running && _imageBytes != null;

    return Scaffold(
      appBar: AppBar(title: const Text('タコグラフ解析（MVP）')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: FilledButton(
                    onPressed: _running ? null : _takePhoto,
                    child: const Text('カメラで撮影'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonal(
                    onPressed: _running ? null : _pickFromGallery,
                    child: const Text('写真を選択'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  border: Border.all(),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: _imageBytes == null
                    ? const Center(child: Text('画像を選択してください'))
                    : Image.memory(_imageBytes!, fit: BoxFit.contain),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Text('チャート種別'),
                const SizedBox(width: 12),
                DropdownButton<String>(
                  value: _chartType,
                  items: const [
                    DropdownMenuItem(value: '24h', child: Text('24時間盤')),
                    DropdownMenuItem(value: '12h', child: Text('12時間盤')),
                  ],
                  onChanged: _running
                      ? null
                      : (v) {
                          if (v == null) return;
                          setState(() {
                            _chartType = v;
                          });
                        },
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text('0時位置補正: ${_midnightOffsetDeg.toStringAsFixed(0)}°'),
            Slider(
              min: -180,
              max: 180,
              divisions: 360,
              value: _midnightOffsetDeg,
              onChanged: _running
                  ? null
                  : (v) {
                      setState(() {
                        _midnightOffsetDeg = v;
                      });
                    },
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: canRun ? _analyze : null,
                child: Text(_running ? '解析中…' : '解析開始'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
