import 'package:flutter/material.dart';

import 'pages/input_page.dart';

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
      home: const InputPage(),
    );
  }
}
