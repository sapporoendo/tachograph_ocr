import 'package:flutter/material.dart';

import 'pages/input_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    const navy = Color(0xFF0B1B3B);
    const orange = Color(0xFFFF6F00);
    return MaterialApp(
      title: 'たこみる',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: navy,
          primary: navy,
          secondary: orange,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: navy,
        ),
      ),
      home: const InputPage(),
    );
  }
}
