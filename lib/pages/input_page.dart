import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../analyzer/analyzer_client.dart';
import '../analyzer/models.dart';
import '../storage/master_store.dart';
import 'error_page.dart';
import 'history_page.dart';
import 'result_page.dart';
import 'settings_page.dart';

class InputPage extends StatefulWidget {
  const InputPage({super.key});

  @override
  State<InputPage> createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  late AnalyzerClient _client;

  static const bool _showDemoToggle = bool.fromEnvironment('SHOW_DEMO_TOGGLE', defaultValue: true);
  static const bool _defaultDemoMode = bool.fromEnvironment('DEMO_MODE', defaultValue: false);

  bool _running = false;
  Uint8List? _imageBytes;
  String _filename = 'tachograph.jpg';
  final String _chartType = '24h';
  bool _demoMode = _defaultDemoMode;

  MasterStore? _store;
  List<String> _drivers = const [];
  List<String> _vehicles = const [];
  String? _selectedDriver;
  String? _selectedVehicle;
  final TextEditingController _distanceKmController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _client = AnalyzerClient(demoMode: _demoMode);
    _loadMasters();
  }

  @override
  void dispose() {
    _distanceKmController.dispose();
    super.dispose();
  }

  Future<void> _loadMasters() async {
    final s = await MasterStore.open();
    final vehicles = s.getVehicles();
    final drivers = s.getDrivers();
    setState(() {
      _store = s;
      _vehicles = vehicles;
      _drivers = drivers;
      if (_selectedVehicle == null || !vehicles.contains(_selectedVehicle)) {
        _selectedVehicle = vehicles.isNotEmpty ? vehicles.first : null;
      }
      if (_selectedDriver == null || !drivers.contains(_selectedDriver)) {
        _selectedDriver = drivers.isNotEmpty ? drivers.first : null;
      }
    });
  }

  Future<void> _selectImage() async {
    if (_running) return;
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.photo_camera, size: 30),
                title: const Text('カメラで撮影', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
                onTap: () => Navigator.of(context).pop(ImageSource.camera),
              ),
              ListTile(
                leading: const Icon(Icons.photo_library, size: 30),
                title: const Text('写真ライブラリ', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
                onTap: () => Navigator.of(context).pop(ImageSource.gallery),
              ),
              const SizedBox(height: 8),
            ],
          ),
        );
      },
    );
    if (source == null) return;
    if (source == ImageSource.camera) {
      await _takePhoto();
    } else {
      await _pickFromGallery();
    }
  }

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
    final driver = _selectedDriver;
    final vehicle = _selectedVehicle;
    if (driver == null || driver.isEmpty || vehicle == null || vehicle.isEmpty) return;

    final distanceText = _distanceKmController.text.trim();
    final distanceKm = distanceText.isEmpty ? null : double.tryParse(distanceText);

    setState(() {
      _running = true;
    });

    try {
      final AnalyzeResult result = await _client.createRecord(
        imageBytes: bytes,
        filename: _filename,
        driverName: driver,
        vehicleNo: vehicle,
        distanceKm: distanceKm,
        chartType: _chartType,
        midnightOffsetDeg: null,
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
    final mastersReady = _store != null;
    final canRun = !_running && _imageBytes != null && (_selectedVehicle?.isNotEmpty ?? false) && (_selectedDriver?.isNotEmpty ?? false);

    return Scaffold(
      appBar: AppBar(
        title: const Text('タコグラフ解析（MVP）'),
        actions: [
          IconButton(
            tooltip: '履歴',
            onPressed: () async {
              await Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => HistoryPage(client: _client)),
              );
            },
            icon: const Icon(Icons.list_alt),
          ),
          IconButton(
            tooltip: '設定',
            onPressed: () async {
              await Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const SettingsPage()),
              );
              await _loadMasters();
            },
            icon: const Icon(Icons.settings),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_showDemoToggle) ...[
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('DEMOモード'),
                value: _demoMode,
                onChanged: _running
                    ? null
                    : (v) {
                        setState(() {
                          _demoMode = v;
                          _client = AnalyzerClient(demoMode: _demoMode);
                        });
                      },
              ),
              const SizedBox(height: 8),
            ],
            if (!mastersReady)
              const Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: LinearProgressIndicator(),
              ),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _selectedDriver,
                    decoration: const InputDecoration(
                      labelText: 'ドライバー',
                      border: OutlineInputBorder(),
                    ),
                    style: const TextStyle(fontSize: 24, color: Colors.black, fontWeight: FontWeight.w700),
                    items: _drivers
                        .map((d) => DropdownMenuItem(value: d, child: Text(d, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700))))
                        .toList(),
                    onChanged: _running
                        ? null
                        : (v) {
                            setState(() {
                              _selectedDriver = v;
                            });
                          },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _selectedVehicle,
                    decoration: const InputDecoration(
                      labelText: '車両番号',
                      border: OutlineInputBorder(),
                    ),
                    style: const TextStyle(fontSize: 24, color: Colors.black, fontWeight: FontWeight.w700),
                    items: _vehicles
                        .map((d) => DropdownMenuItem(value: d, child: Text(d, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700))))
                        .toList(),
                    onChanged: _running
                        ? null
                        : (v) {
                            setState(() {
                              _selectedVehicle = v;
                            });
                          },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            TextField(
              controller: _distanceKmController,
              enabled: !_running,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
              decoration: const InputDecoration(
                labelText: '走行距離 (km)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: InkWell(
                onTap: _selectImage,
                borderRadius: BorderRadius.circular(14),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEFEFEF),
                    border: Border.all(color: const Color(0xFF444444), width: 2),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: _imageBytes == null
                      ? const Center(
                          child: Text(
                            'タップで撮影・選択',
                            style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800),
                          ),
                        )
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: Image.memory(_imageBytes!, fit: BoxFit.contain),
                        ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            if (!canRun)
              const Padding(
                padding: EdgeInsets.only(bottom: 6),
                child: Text(
                  '画像 + ドライバー + 車両番号 を選択すると解析できます',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF444444)),
                ),
              ),
            SizedBox(
              width: double.infinity,
              height: 68,
              child: FilledButton(
                style: FilledButton.styleFrom(
                  backgroundColor: const Color(0xFFFF6F00),
                  disabledBackgroundColor: const Color(0xFFCCCCCC),
                ),
                onPressed: canRun ? _analyze : null,
                child: Text(
                  _running ? '解析中…' : '解析開始',
                  style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w900),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
