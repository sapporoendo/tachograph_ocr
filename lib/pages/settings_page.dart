import 'package:flutter/material.dart';

import '../storage/master_store.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  MasterStore? _store;
  List<String> _vehicles = const [];
  List<String> _drivers = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final s = await MasterStore.open();
    setState(() {
      _store = s;
      _vehicles = s.getVehicles();
      _drivers = s.getDrivers();
    });
  }

  Future<String?> _promptText({
    required String title,
    String initialValue = '',
  }) async {
    final controller = TextEditingController(text: initialValue);
    return showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text(title),
          content: TextField(
            controller: controller,
            autofocus: true,
            style: const TextStyle(fontSize: 20),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('キャンセル'),
            ),
            FilledButton(
              onPressed: () =>
                  Navigator.of(context).pop(controller.text.trim()),
              child: const Text('保存'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _addVehicle() async {
    final v = await _promptText(title: '車両番号を追加');
    if (v == null || v.isEmpty) return;
    final s = _store;
    if (s == null) return;
    await s.addVehicle(v);
    await _load();
  }

  Future<void> _addDriver() async {
    final v = await _promptText(title: 'ドライバー名を追加');
    if (v == null || v.isEmpty) return;
    final s = _store;
    if (s == null) return;
    await s.addDriver(v);
    await _load();
  }

  Future<void> _renameVehicle(String from) async {
    final to = await _promptText(title: '車両番号を編集', initialValue: from);
    if (to == null || to.isEmpty || to == from) return;
    final s = _store;
    if (s == null) return;
    await s.renameVehicle(from, to);
    await _load();
  }

  Future<void> _renameDriver(String from) async {
    final to = await _promptText(title: 'ドライバー名を編集', initialValue: from);
    if (to == null || to.isEmpty || to == from) return;
    final s = _store;
    if (s == null) return;
    await s.renameDriver(from, to);
    await _load();
  }

  Future<void> _deleteVehicle(String v) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('削除しますか？'),
        content: Text('車両番号: $v'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('キャンセル'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('削除'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    final s = _store;
    if (s == null) return;
    await s.removeVehicle(v);
    await _load();
  }

  Future<void> _deleteDriver(String v) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('削除しますか？'),
        content: Text('ドライバー: $v'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('キャンセル'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('削除'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    final s = _store;
    if (s == null) return;
    await s.removeDriver(v);
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _store != null;

    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: Image.asset(
            'assets/images/takomiru_logo.png',
            height: 28,
            fit: BoxFit.contain,
          ),
          bottom: const TabBar(
            tabs: [
              Tab(text: '車両'),
              Tab(text: 'ドライバー'),
            ],
          ),
        ),
        body: !ready
            ? const Center(child: CircularProgressIndicator())
            : TabBarView(
                children: [
                  _buildList(
                    title: '車両番号',
                    items: _vehicles,
                    onAdd: _addVehicle,
                    onRename: _renameVehicle,
                    onDelete: _deleteVehicle,
                  ),
                  _buildList(
                    title: 'ドライバー',
                    items: _drivers,
                    onAdd: _addDriver,
                    onRename: _renameDriver,
                    onDelete: _deleteDriver,
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildList({
    required String title,
    required List<String> items,
    required Future<void> Function() onAdd,
    required Future<void> Function(String) onRename,
    required Future<void> Function(String) onDelete,
  }) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              FilledButton.icon(
                onPressed: onAdd,
                icon: const Icon(Icons.add),
                label: const Text('追加', style: TextStyle(fontSize: 18)),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: ListView.separated(
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (context, i) {
              final v = items[i];
              return ListTile(
                title: Text(
                  v,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                subtitle: const Text('タップで編集', style: TextStyle(fontSize: 14)),
                onTap: () => onRename(v),
                trailing: IconButton(
                  iconSize: 30,
                  onPressed: () => onDelete(v),
                  icon: const Icon(Icons.delete_outline),
                ),
                minVerticalPadding: 16,
              );
            },
          ),
        ),
      ],
    );
  }
}
