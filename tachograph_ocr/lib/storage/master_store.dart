// Copyright (c) 2026 Kumiko Naito. All rights reserved.
import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class MasterStore {
  static const _keyVehicles = 'master_vehicles';
  static const _keyDrivers = 'master_drivers';

  static const List<String> defaultVehicles = [
    '2600',
    '2541',
    '2432',
    '2469',
    '2436',
  ];

  static const List<String> defaultDrivers = [
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
  ];

  final SharedPreferences _prefs;

  MasterStore._(this._prefs);

  static Future<MasterStore> open() async {
    final prefs = await SharedPreferences.getInstance();
    final store = MasterStore._(prefs);
    await store._ensureDefaults();
    return store;
  }

  Future<void> _ensureDefaults() async {
    if (!_prefs.containsKey(_keyVehicles)) {
      await setVehicles(defaultVehicles);
    }
    if (!_prefs.containsKey(_keyDrivers)) {
      await setDrivers(defaultDrivers);
    }
  }

  List<String> getVehicles() {
    final raw = _prefs.getString(_keyVehicles);
    if (raw == null || raw.isEmpty) {
      return List<String>.from(defaultVehicles);
    }
    final decoded = jsonDecode(raw);
    if (decoded is List) {
      return decoded.map((e) => e.toString()).where((e) => e.isNotEmpty).toSet().toList()..sort();
    }
    return List<String>.from(defaultVehicles);
  }

  List<String> getDrivers() {
    final raw = _prefs.getString(_keyDrivers);
    if (raw == null || raw.isEmpty) {
      return List<String>.from(defaultDrivers);
    }
    final decoded = jsonDecode(raw);
    if (decoded is List) {
      final list = decoded.map((e) => e.toString()).where((e) => e.isNotEmpty).toSet().toList()..sort();
      if (list.isEmpty) {
        return List<String>.from(defaultDrivers);
      }
      return list;
    }
    return List<String>.from(defaultDrivers);
  }

  Future<void> setVehicles(List<String> vehicles) async {
    final uniq = vehicles.map((e) => e.trim()).where((e) => e.isNotEmpty).toSet().toList()..sort();
    await _prefs.setString(_keyVehicles, jsonEncode(uniq));
  }

  Future<void> setDrivers(List<String> drivers) async {
    final uniq = drivers.map((e) => e.trim()).where((e) => e.isNotEmpty).toSet().toList()..sort();
    await _prefs.setString(_keyDrivers, jsonEncode(uniq));
  }

  Future<void> addVehicle(String v) async {
    final list = getVehicles();
    list.add(v);
    await setVehicles(list);
  }

  Future<void> removeVehicle(String v) async {
    final list = getVehicles()..removeWhere((e) => e == v);
    await setVehicles(list);
  }

  Future<void> addDriver(String name) async {
    final list = getDrivers();
    list.add(name);
    await setDrivers(list);
  }

  Future<void> removeDriver(String name) async {
    final list = getDrivers()..removeWhere((e) => e == name);
    await setDrivers(list);
  }

  Future<void> renameVehicle(String from, String to) async {
    final list = getVehicles();
    final idx = list.indexOf(from);
    if (idx < 0) return;
    list[idx] = to;
    await setVehicles(list);
  }

  Future<void> renameDriver(String from, String to) async {
    final list = getDrivers();
    final idx = list.indexOf(from);
    if (idx < 0) return;
    list[idx] = to;
    await setDrivers(list);
  }
}
