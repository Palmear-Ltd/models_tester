// =============================================================================
// palmear_mobile_processor.dart
// =============================================================================
// ISOLATED SCAN PIPELINE for the data science team.
//
// This file is a stand-alone extract of the mobile app's scan path:
//   1) Preprocessing  – chunk FIFO, optional bandpass, log-mel spectrogram,
//                       feature crop, scaler normalization, energy gate
//   2) Model calling  – TensorFlow Lite inference on each 2.5s window
//   3) Postprocessing – per-chunk score threshold → count → session verdict
//
// Source files (DO NOT USE THIS FILE IN THE APP BUILD – reference only):
//   lib/palmear_ai/audio_processor/new_audio_processor.dart   (orchestration)
//   lib/palmear_ai/audio_processor/chunk_fifo_buffer.dart
//   lib/palmear_ai/preprocessor/energy_processor.dart
//   lib/palmear_ai/preprocessor/preprocessor_utils.dart       (normalize)
//   lib/palmear_ai/mel_spectrogram/audio_dsp.dart
//   lib/palmear_ai/mel_spectrogram/new_mel_spectogram.dart
//   lib/palmear_ai/tf_lite_predictor/tf_lite_predictor.dart
//   lib/palmear_ai/post_processor/new_postprocessor.dart
//   lib/palmear_ai/schema/models/{model_params,scaler,new_postprocess_output}.dart
//
// External packages used by this pipeline:
//   - package:fftea/fftea.dart          (FFT for power spectrum)
//   - package:tflite_flutter/tflite_flutter.dart
//
// Scan timing (as in the app):
//   sampleRate          = 44100 Hz
//   hop / "frame"       = 0.5 s  → 22050 samples (PCM16 mono)
//   model window        = 2.5 s  → 5 hops via ChunkFifoBuffer (maxChunks=5)
//   session length      = detectionWinSize seconds (default 20)
//   total inferences    = detectionWinSize * 2  (e.g. 40 scores)
//
// Per-chunk inference flow (isolate entry point):
//   audio[2.5s]
//     → optional bandpass 500–8000 Hz @ 44100
//     → log-mel spectrogram [33 mels × ~98 frames]
//     → transpose → drop last mel bin → [~98 × 32]
//     → (x - mean) / sqrt(var)   using model Scaler
//     → TFLite → scalar score in [0, 1] (typical sigmoid output)
//
// Session postprocess:
//   algoTriggers = count( score_i >= labelScore )
//   if algoTriggers >= sumInfThr → "detected"
//   else if algoTriggers >= sumSusThr → "suspicious"
//   else → "notDetected"
// =============================================================================

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fftea/fftea.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// Set true while experimenting offline to print intermediate shapes/values.
const bool kPipelineDebug = false;

// =============================================================================
// SECTION 0 – High-level API (matches app scan semantics)
// =============================================================================

/// Parameters that accompany each TFLite model (from model caliboration / Firestore).
class ModelParams {
  /// Energy gate threshold for [EnergyProcessor] (default ~20/sqrt(10)).
  final double energyThr;

  /// Minimum positive-chunk count for "detected".
  final double sumInfThr;

  /// Minimum positive-chunk count for "suspicious".
  final double sumSusThr;

  /// Per-chunk score threshold (e.g. 0.6).
  final double labelScore;

  /// Session length in seconds. Chunk count ≈ detectionWinSize * 2.
  final double detectionWinSize;

  /// UI flag: show/hide live suspicious indicator.
  final bool susIndicator;

  ModelParams({
    required this.energyThr,
    required this.sumInfThr,
    required this.sumSusThr,
    required this.labelScore,
    required this.detectionWinSize,
    required this.susIndicator,
  });

  factory ModelParams.fromMap(Map<String, dynamic> map) {
    return ModelParams(
      energyThr: map['energyThr'].toDouble(),
      sumInfThr: map['sumInfThr'].toDouble(),
      sumSusThr: map['sumSusThr'].toDouble(),
      labelScore: map['labelScore'].toDouble(),
      detectionWinSize: map['detectionWinSize'].toDouble(),
      susIndicator: map['susIndicator'] as bool,
    );
  }

  Map<String, dynamic> toMap() => {
        'energyThr': energyThr,
        'sumInfThr': sumInfThr,
        'sumSusThr': sumSusThr,
        'labelScore': labelScore,
        'detectionWinSize': detectionWinSize,
        'susIndicator': susIndicator,
      };
}

/// Feature-wise mean / variance used to normalize the mel matrix before TFLite.
class Scaler {
  final List<double> means;
  final List<double> vars;

  Scaler({required this.means, required this.vars});

  Map<String, dynamic> toMap() => {'mean': means, 'var': vars};

  factory Scaler.fromMap(Map<String, dynamic> map) {
    final meanList = map['mean'].map((e) => e.toDouble()).toList();
    final varList = map['var'].map((e) => e.toDouble()).toList();
    return Scaler(
      means: List<double>.from(meanList),
      vars: List<double>.from(varList),
    );
  }

  factory Scaler.fromJson(String source) =>
      Scaler.fromMap(json.decode(source) as Map<String, dynamic>);
}

/// Output of session-level postprocessing.
class PostProcessOutput {
  final int algoTriggers;
  final String result; // "detected" | "suspicious" | "notDetected"

  PostProcessOutput({required this.algoTriggers, required this.result});

  @override
  String toString() =>
      'PostProcessOutput(algoTriggers: $algoTriggers, result: $result)';
}

/// End-to-end helpers mirroring what the app does while scanning.
class PalmearMobileProcessor {
  static const int sampleRate = 44100;
  static const int frameSize = 22050; // 0.5 s @ 44100
  static const int fifoChunks = 5; // → 2.5 s flattened window

  /// Runs **one** inference window (exactly what each isolate does).
  ///
  /// [windowAudio] must be ~2.5 s mono floats in [-1, 1]
  /// (length ≈ [ChunkFifoBuffer.flattened] = 5 * 22050 = 110250).
  static Future<double> inferChunk({
    required List<double> windowAudio,
    required String modelFilePath,
    required Scaler scaler,
    bool applyBandpassFilter = false,
  }) async {
    var signal = List<double>.from(windowAudio);

    // --- PREPROCESS: optional bandpass (app defaults: 500–8000 Hz) ----------
    if (applyBandpassFilter) {
      signal = AudioDsp.applyBandpassFilter(
        signal,
        lowcut: 500,
        highcut: 8000,
        fs: sampleRate.toDouble(),
      );
    }

    // --- PREPROCESS: log-mel [nMels × nFrames] ------------------------------
    final List<List<double>> logMel = await computeMelSpectrogram(signal);

    // Transpose → [nFrames × nMels], drop last mel bin → [nFrames × 32]
    final transposed = logMel.transpose();
    final features98x32 =
        transposed.map((r) => r.sublist(0, r.length - 1)).toList();

    // --- PREPROCESS: scaler normalization -----------------------------------
    final normFeatures = PreprocessorUtils.normalize(
      features98x32,
      mean: scaler.means,
      variance: scaler.vars,
    );

    // --- MODEL: TFLite ------------------------------------------------------
    final predictor = TFLitePredictor(modelFilePath);
    try {
      final output = predictor.predict(normFeatures.flatten());
      final score = (output[0][0] as num).toDouble();
      if (kPipelineDebug) {
        // ignore: avoid_print
        print('inference=$score  features=${features98x32.length}x'
            '${features98x32.isEmpty ? 0 : features98x32[0].length}');
      }
      return score;
    } finally {
      predictor.close();
    }
  }

  /// Session postprocess over all chunk scores collected during a scan.
  static PostProcessOutput postprocessSession({
    required ModelParams params,
    required List<double> chunkScores,
  }) {
    return PostProcessor().processLabels(params, chunkScores);
  }

  /// Convenience: process a list of successive 0.5 s hops the way the app does.
  ///
  /// [halfSecondChunks] are consecutive 22050-sample buffers.
  /// A FIFO of 5 hops builds each 2.5 s window; after [params.detectionWinSize]*2
  /// windows, scores are aggregated.
  static Future<PostProcessOutput> runScanFromHalfSecondChunks({
    required List<List<double>> halfSecondChunks,
    required String modelFilePath,
    required Scaler scaler,
    required ModelParams params,
    bool applyBandpassFilter = false,
    void Function(double score)? onChunkScore,
  }) async {
    final fifo = ChunkFifoBuffer(frameSize: frameSize, maxChunks: fifoChunks);
    final energy = EnergyProcessor(energyThreshold: params.energyThr);
    final scores = <double>[];
    final totalWindows = params.detectionWinSize.toInt() * 2;

    for (final hop in halfSecondChunks) {
      if (scores.length >= totalWindows) break;
      if (hop.length != frameSize) {
        throw ArgumentError(
          'Each hop must be length $frameSize (0.5s), got ${hop.length}',
        );
      }
      fifo.addChunk(hop);

      // Energy gate (app aborts the scan if high energy and not ignored).
      if (energy.isHighEnergy(fifo.flattened)) {
        // DS note: app sets isHighEnergy and stops; here we still continue
        // but mark via debug so offline evaluation can observe it.
        if (kPipelineDebug) {
          // ignore: avoid_print
          print('HIGH ENERGY at window ${scores.length + 1}');
        }
      }

      final score = await inferChunk(
        windowAudio: fifo.flattened,
        modelFilePath: modelFilePath,
        scaler: scaler,
        applyBandpassFilter: applyBandpassFilter,
      );
      scores.add(score);
      onChunkScore?.call(score);
    }

    return postprocessSession(params: params, chunkScores: scores);
  }
}

// =============================================================================
// SECTION 1 – Chunk FIFO (builds the 2.5 s model window from 0.5 s hops)
// =============================================================================

/// Fixed-length FIFO of audio hops. Pre-filled with zeros; after [maxChunks]
/// real adds it behaves as a normal FIFO (evict oldest).
class ChunkFifoBuffer {
  final int frameSize;
  final int maxChunks;
  final List<_FifoSlot> _queue = [];
  int _filledCount = 0;

  ChunkFifoBuffer({required this.frameSize, this.maxChunks = 5}) {
    _reset();
  }

  void _reset() {
    _queue
      ..clear()
      ..addAll(List.generate(
        maxChunks,
        (_) => _FifoSlot(List<double>.filled(frameSize, 0.0)),
      ));
    _filledCount = 0;
  }

  void addChunk(List<double> chunk) {
    if (chunk.length != frameSize) {
      throw ArgumentError(
        'Must have length frameSize ($frameSize), got ${chunk.length}',
      );
    }
    _queue.removeAt(0);
    _queue.add(_FifoSlot(List<double>.from(chunk)));
    _filledCount = math.min(_filledCount + 1, maxChunks);
  }

  List<List<double>> get chunks =>
      List.unmodifiable(_queue.map((s) => s.data));

  /// Concatenated samples: length == maxChunks * frameSize (e.g. 110250).
  List<double> get flattened =>
      _queue.expand((s) => s.data).toList(growable: false);

  void clear() => _reset();
}

class _FifoSlot {
  final List<double> data;
  _FifoSlot(this.data);
}

// =============================================================================
// SECTION 2 – Energy gate (preprocessing / quality control)
// =============================================================================

class EnergyProcessor {
  final int energyBufferSize;
  final double winSize;
  final double hopSize;
  late final double overlapFactor;
  late final double effectiveEnBuffLen;
  final double energyThreshold;
  late List<double> energyVec;

  EnergyProcessor({
    this.energyBufferSize = 10,
    this.winSize = 2.5,
    this.hopSize = 0.5,
    double? energyThreshold,
  }) : energyThreshold = energyThreshold ?? (20 / math.sqrt(10)) {
    overlapFactor = winSize / hopSize;
    effectiveEnBuffLen = hopSize * energyBufferSize;
    energyVec = List.filled(energyBufferSize, 0.0);
  }

  /// energy = sum(x^2) / (overlapFactor * effectiveEnBuffLen)
  /// energyAll = sqrt(sum(energyVec)); returns true if energyAll > threshold.
  bool isHighEnergy(List<double> audioBuffer) {
    for (int i = energyBufferSize - 1; i > 0; i--) {
      energyVec[i] = energyVec[i - 1];
    }
    final sumSquares = audioBuffer.fold(0.0, (p, x) => p + x * x);
    final energy = sumSquares / (overlapFactor * effectiveEnBuffLen);
    energyVec[0] = energy;
    final energyAll = math.sqrt(energyVec.reduce((a, b) => a + b));
    if (kPipelineDebug) {
      // ignore: avoid_print
      print('energyThr=$energyThreshold energyAll=$energyAll');
    }
    return energyAll > energyThreshold;
  }

  void reset() {
    for (var i = 0; i < energyVec.length; i++) {
      energyVec[i] = 0.0;
    }
  }
}

// =============================================================================
// SECTION 3 – Bandpass (optional preprocess before mel)
// =============================================================================

class AudioDsp {
  /// SciPy-equivalent butterworth bandpass via DF-II transposed lfilter.
  /// Precomputed coeffs only match fs=44100, lowcut=500, highcut=8000, order=4
  /// (exact coeffs used by the app isolate).
  static List<double> applyBandpassFilter(
    List<double> x, {
    required double lowcut,
    required double highcut,
    required double fs,
    int order = 4,
    List<double>? b,
    List<double>? a,
  }) {
    if (lowcut <= 0 || highcut <= 0 || fs <= 0) {
      throw ArgumentError('lowcut/highcut/fs must be > 0');
    }
    if (highcut >= fs / 2) {
      throw ArgumentError('highcut must be < Nyquist (fs/2)');
    }
    if (lowcut >= highcut) {
      throw ArgumentError('lowcut must be < highcut');
    }

    final coeffs = (b != null && a != null)
        ? _BpCoeffs(b, a)
        : _butterBandpassPrecomputed(
            lowcut: lowcut, highcut: highcut, fs: fs, order: order);

    return _lfilterDf2T(x, coeffs.b, coeffs.a);
  }

  static List<double> _lfilterDf2T(
      List<double> x, List<double> bIn, List<double> aIn) {
    if (aIn.isEmpty || bIn.isEmpty) throw ArgumentError('Empty coefficients');
    var a0 = aIn[0];
    if (a0 == 0.0) throw ArgumentError('a[0] must be non-zero');

    final b = List<double>.from(bIn);
    final a = List<double>.from(aIn);
    if (a0 != 1.0) {
      for (var i = 0; i < b.length; i++) {
        b[i] /= a0;
      }
      for (var i = 0; i < a.length; i++) {
        a[i] /= a0;
      }
      a0 = 1.0;
    }

    final n = math.max(a.length, b.length) - 1;
    if (b.length < n + 1) b.addAll(List.filled(n + 1 - b.length, 0.0));
    if (a.length < n + 1) a.addAll(List.filled(n + 1 - a.length, 0.0));

    final y = Float64List(x.length);
    final z = Float64List(n);

    for (var k = 0; k < x.length; k++) {
      final xv = x[k].toDouble();
      double yv = b[0] * xv + (n > 0 ? z[0] : 0.0);
      for (var i = 1; i < n; i++) {
        z[i - 1] = b[i] * xv + z[i] - a[i] * yv;
      }
      if (n > 0) {
        z[n - 1] = b[n] * xv - a[n] * yv;
      }
      y[k] = yv;
    }
    return y.toList(growable: false);
  }

  static _BpCoeffs _butterBandpassPrecomputed({
    required double lowcut,
    required double highcut,
    required double fs,
    required int order,
  }) {
    const tol = 1e-9;
    final match = (order == _BpCoeffs.order) &&
        (fs - _BpCoeffs.fs).abs() < tol &&
        (lowcut - _BpCoeffs.lowCut).abs() < tol &&
        (highcut - _BpCoeffs.highcut).abs() < tol;

    if (!match) {
      throw UnsupportedError(
        'Unsupported bandpass params. Export SciPy b/a and pass explicitly.',
      );
    }

    // scipy.signal.butter(4, [500/(0.5*44100), 8000/(0.5*44100)], btype="band")
    const b = <double>[
      0.02782219133158324,
      0,
      -0.11128876532633296,
      0,
      0.16693314798949943,
      0,
      -0.11128876532633296,
      0,
      0.02782219133158324,
    ];
    const a = <double>[
      1,
      -5.0183594408704062,
      11.059767387158189,
      -14.257305019489259,
      11.975233504014817,
      -6.7458600840216985,
      2.459405225076718,
      -0.52516968964355892,
      0.052302554816541225,
    ];
    return const _BpCoeffs(b, a);
  }
}

class _BpCoeffs {
  static const double highcut = 8000;
  static const double lowCut = 500;
  static const double fs = 44100.0;
  static const int order = 4;
  final List<double> b;
  final List<double> a;
  const _BpCoeffs(this.b, this.a);
}

// =============================================================================
// SECTION 4 – Log-mel spectrogram (core feature preprocess)
// =============================================================================

/// Python banker's rounding (half-to-even) for frame length parity with training.
int pyRound(double x, [double eps = 1e-8]) {
  final int lo = x.floor();
  final double frac = x - lo;
  if ((frac - 0.5).abs() < eps) {
    return (lo % 2 == 0) ? lo : lo + 1;
  } else if (frac < 0.5) {
    return lo;
  } else {
    return lo + 1;
  }
}

List<double> hamming(int M, {bool sym = true}) {
  return _generalHamming(M, 0.54, sym: sym);
}

List<double> _generalHamming(int M, double alpha, {bool sym = true}) {
  if (M < 0) throw ArgumentError.value(M, 'M', 'must be non-negative');
  return _generalCosineImpl(M, [alpha, 1.0 - alpha], sym: sym);
}

List<double> _generalCosineImpl(int M, List<double> a, {bool sym = true}) {
  if (M <= 1) return List<double>.filled(M, 1.0);
  final int mext = sym ? M : M + 1;
  final bool needsTrunc = !sym;
  final fac = List<double>.generate(mext, (i) {
    return -math.pi + 2 * math.pi * i / (mext - 1);
  });
  final w = List<double>.filled(mext, 0.0);
  for (var k = 0; k < a.length; k++) {
    for (var n = 0; n < mext; n++) {
      w[n] += a[k] * math.cos(k * fac[n]);
    }
  }
  return needsTrunc ? w.sublist(0, w.length - 1) : w;
}

/// Log-mel spectrogram of a mono float signal.
/// Returns [numMels × numFrames] (default 33 × ~98 for 2.5 s @ 44100).
Future<List<List<double>>> computeMelSpectrogram(
  List<double> audio, {
  int fs = 44100,
  int numMels = 33,
  double fmin = 50.0,
  double fmax = 10000.0,
}) async {
  final windowLength = pyRound(0.050 * fs); // 50 ms
  final fttLength = math.max(1024, windowLength);
  final hopLength = pyRound(0.025 * fs); // 25 ms

  final melFilterBank = melFilter(
    sr: fs.toDouble(),
    nFft: fttLength,
    nMels: numMels,
    fMin: fmin,
    fMax: fmax,
    norm: 'slaney',
  );

  final window = hamming(windowLength, sym: true);
  final frames = _frameAndWindow(
    signal: audio,
    frameLength: windowLength,
    hopLength: hopLength,
    window: window,
  );

  final powerSpectrum = computePowerSpectrumFromFrames(
    frames: frames,
    fftLength: fttLength,
    window: window,
  );

  final int nFrames = powerSpectrum.length;
  final int nFreqs = powerSpectrum[0].length;
  final transposedPS = List<List<double>>.generate(
    nFreqs,
    (i) => List<double>.generate(nFrames, (j) => powerSpectrum[j][i]),
  );

  final int nMels = melFilterBank.length;
  final int nFreqs0 = melFilterBank[0].length;
  final melSpec = List<List<double>>.generate(nMels, (i) {
    return List<double>.generate(nFrames, (j) {
      double sum = 0.0;
      for (int k = 0; k < nFreqs0; k++) {
        sum += melFilterBank[i][k] * transposedPS[k][j];
      }
      return sum;
    });
  });

  // Floor from 5th percentile of positive energies (training parity).
  final positives = <double>[];
  for (final row in melSpec) {
    for (final v in row) {
      if (v > 0) positives.add(v);
    }
  }
  double minVal;
  if (positives.isNotEmpty) {
    positives.sort();
    final rank = 0.05 * (positives.length - 1);
    final lo = rank.floor();
    final hi = rank.ceil();
    if (lo == hi) {
      minVal = positives[lo];
    } else {
      final w = rank - lo;
      minVal = positives[lo] * (1 - w) + positives[hi] * w;
    }
  } else {
    minVal = 1e-12;
  }

  const eps = 1e-12;
  final thresh = minVal * 1e-3;
  return List<List<double>>.generate(nMels, (i) {
    return List<double>.generate(nFrames, (j) {
      final v = melSpec[i][j] < thresh ? thresh : melSpec[i][j];
      return 10 * math.log(v + eps) / math.ln10;
    });
  });
}

List<List<double>> _frameAndWindow({
  required List<double> signal,
  required int frameLength,
  required int hopLength,
  required List<double> window,
}) {
  final n = signal.length;
  if (n < frameLength) {
    throw ArgumentError('signal length $n < frameLength $frameLength');
  }
  if (hopLength < 1) throw ArgumentError('hopLength must be >= 1');
  if (window.length != frameLength) {
    throw ArgumentError('window length must equal frameLength');
  }

  final nFrames = ((n - frameLength + 1) / hopLength).floor();
  final frames = List<List<double>>.generate(nFrames, (i) {
    final start = i * hopLength;
    return signal.sublist(start, start + frameLength);
  });
  for (var i = 0; i < nFrames; i++) {
    final row = frames[i];
    for (var j = 0; j < frameLength; j++) {
      row[j] *= window[j];
    }
  }
  return frames;
}

List<List<double>> computePowerSpectrumFromFrames({
  required List<List<double>> frames,
  required int fftLength,
  required List<double> window,
}) {
  final fft = FFT(fftLength);
  final winSum = window.fold(0.0, (a, b) => a + b);
  final denom = winSum * winSum;

  return frames.map((frame) {
    final spectrum = fft.realFft(frame);
    return List<double>.generate(spectrum.length, (i) {
      final c = spectrum[i];
      return (c.x * c.x + c.y * c.y) / denom;
    });
  }).toList();
}

List<List<double>> melFilter({
  required double sr,
  required int nFft,
  int nMels = 128,
  double fMin = 0.0,
  double? fMax,
  bool htk = false,
  String norm = 'slaney',
}) {
  final fMaxVal = fMax ?? sr / 2;
  final nFreqs = 1 + (nFft ~/ 2);
  final weights = List<List<double>>.generate(
    nMels,
    (_) => List<double>.filled(nFreqs, 0.0),
  );

  final fftFreqs = _fftFrequencies(sr: sr, nFft: nFft);
  final melF = _melFrequencies(nMels + 2, fMin: fMin, fMax: fMaxVal, htk: htk);
  final fdiff = List<double>.generate(
    melF.length - 1,
    (i) => melF[i + 1] - melF[i],
  );
  final ramps = List<List<double>>.generate(
    melF.length,
    (i) => List<double>.generate(fftFreqs.length, (j) => melF[i] - fftFreqs[j]),
  );

  for (int i = 0; i < nMels; i++) {
    for (int j = 0; j < nFreqs; j++) {
      final lower = -ramps[i][j] / fdiff[i];
      final upper = ramps[i + 2][j] / fdiff[i + 1];
      weights[i][j] = math.max(0.0, math.min(lower, upper));
    }
  }

  if (norm == 'slaney') {
    for (int i = 0; i < nMels; i++) {
      final enorm = 2.0 / (melF[i + 2] - melF[i]);
      for (int j = 0; j < nFreqs; j++) {
        weights[i][j] *= enorm;
      }
    }
  } else {
    throw ArgumentError('Unsupported norm: $norm');
  }
  return weights;
}

List<double> _fftFrequencies({required double sr, required int nFft}) {
  final nFreqs = 1 + (nFft ~/ 2);
  return List<double>.generate(nFreqs, (i) => i * (sr / nFft));
}

List<double> _melFrequencies(
  int nMels, {
  required double fMin,
  required double fMax,
  bool htk = false,
}) {
  final minMel = _hzToMel(fMin, htk: htk);
  final maxMel = _hzToMel(fMax, htk: htk);
  final mels = List<double>.generate(
    nMels,
    (i) => minMel + (maxMel - minMel) * i / (nMels - 1),
  );
  return mels.map((mel) => _melToHz(mel, htk: htk)).toList();
}

double _hzToMel(double freq, {bool htk = false}) {
  if (htk) return 2595.0 * math.log(1.0 + freq / 700.0) / math.ln10;
  const fMin = 0.0;
  const fSp = 200.0 / 3;
  final mel = (freq - fMin) / fSp;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  final logStep = math.log(6.4) / 27.0;
  if (freq >= minLogHz) {
    return minLogMel + math.log(freq / minLogHz) / logStep;
  }
  return mel;
}

double _melToHz(double mel, {bool htk = false}) {
  if (htk) return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);
  const fMin = 0.0;
  const fSp = 200.0 / 3;
  final freqs = fMin + fSp * mel;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  final logStep = math.log(6.4) / 27.0;
  if (mel >= minLogMel) {
    return minLogHz * math.exp(logStep * (mel - minLogMel));
  }
  return freqs;
}

// =============================================================================
// SECTION 5 – Scaler normalization
// =============================================================================

class PreprocessorUtils {
  /// Column-wise (x - mean) / sqrt(variance). Zero std → 1 to avoid /0.
  static List<List<double>> normalize(
    List<List<double>> features, {
    List<double>? mean,
    List<double>? variance,
  }) {
    if (features.isEmpty) return [];

    final nSamples = features.length;
    final nFeatures = features[0].length;
    for (final row in features) {
      if (row.length != nFeatures) {
        throw ArgumentError('All feature rows must have the same length.');
      }
    }

    if (mean == null || mean.length != nFeatures) {
      mean = List<double>.filled(nFeatures, 0.0);
      for (var j = 0; j < nFeatures; j++) {
        var sum = 0.0;
        for (var i = 0; i < nSamples; i++) {
          sum += features[i][j];
        }
        mean[j] = sum / nSamples;
      }
    }

    final std = List<double>.filled(nFeatures, 0.0);
    if (variance == null || variance.length != nFeatures) {
      for (var j = 0; j < nFeatures; j++) {
        var sumSq = 0.0;
        for (var i = 0; i < nSamples; i++) {
          final d = features[i][j] - mean[j];
          sumSq += d * d;
        }
        std[j] = math.sqrt(sumSq / nSamples);
        if (std[j] == 0.0) std[j] = 1.0;
      }
    } else {
      for (var j = 0; j < nFeatures; j++) {
        std[j] = math.sqrt(variance[j]);
        if (std[j] == 0.0) std[j] = 1.0;
      }
    }

    return List.generate(
      nSamples,
      (i) => List.generate(
        nFeatures,
        (j) => (features[i][j] - mean![j]) / std[j],
      ),
    );
  }
}

extension _MatrixExtension<T extends num> on List<List<T>> {
  List<List<T>> transpose() {
    if (isEmpty || this[0].isEmpty) return [];
    return List.generate(this[0].length, (i) {
      return List.generate(length, (j) => this[j][i]);
    });
  }
}

extension _Flatten2D<T> on List<List<T>> {
  List<T> flatten() => expand((row) => row).toList();
}

// =============================================================================
// SECTION 6 – TFLite model calling
// =============================================================================

class TFLitePredictor {
  late Interpreter _interpreter;

  TFLitePredictor(String modelPath) {
    _interpreter = Interpreter.fromFile(File(modelPath));
  }

  /// [flatWindow] = flattened [height × 32] mel features (row-major).
  /// Reshaped to [1, expectedHeight, 32, 1] to match the model's input tensor.
  List<dynamic> predict(List<double> flatWindow) {
    assert(flatWindow.length % 32 == 0,
        'Input list length must be divisible by 32');

    final originalInputShape = _interpreter.getInputTensor(0).shape;
    final expectedHeight = originalInputShape[1];
    const width = 32;

    final computedHeight = flatWindow.length ~/ width;
    var window2D = List.generate(
      computedHeight,
      (i) => List.generate(width, (j) => flatWindow[i * width + j]),
    );

    if (computedHeight < expectedHeight) {
      final padRows = expectedHeight - computedHeight;
      for (var i = 0; i < padRows; i++) {
        window2D.add(List.filled(width, 0.0));
      }
    } else if (computedHeight > expectedHeight) {
      window2D = window2D.sublist(0, expectedHeight);
    }

    final input = [
      List.generate(
        expectedHeight,
        (i) => List.generate(width, (j) => [window2D[i][j]]),
      )
    ];

    _interpreter.resizeInputTensor(0, [1, expectedHeight, width, 1]);
    _interpreter.allocateTensors();

    final outputShape = _interpreter.getOutputTensor(0).shape;
    final output = List.generate(
      outputShape[0],
      (i) => List.filled(outputShape[1], 0.0),
    );

    _interpreter.run(input, output);
    return output;
  }

  void close() => _interpreter.close();
}

// =============================================================================
// SECTION 7 – Session postprocessing (verdict from chunk scores)
// =============================================================================

class PostProcessor {
  /// Hard-threshold each chunk score at [ModelParams.labelScore], count hits,
  /// then map count → detected / suspicious / notDetected.
  PostProcessOutput processLabels(ModelParams params, List<double> labels) {
    final labelScore = params.labelScore;
    final infectedThreshold = params.sumInfThr;
    final suspiciousThreshold = params.sumSusThr;

    final scores = labels.map((label) => label >= labelScore).toList();
    final algoTriggers = scores.fold<int>(0, (s, e) => s + (e ? 1 : 0));

    return PostProcessOutput(
      algoTriggers: algoTriggers,
      result: _getResult(algoTriggers, suspiciousThreshold, infectedThreshold),
    );
  }

  String _getResult(
    int trues,
    double suspiciousThreshold,
    double infectedThreshold,
  ) {
    if (trues >= infectedThreshold) return 'detected';
    if (trues >= suspiciousThreshold) return 'suspicious';
    return 'notDetected';
  }
}

// =============================================================================
// SECTION 8 – PCM helpers (mic → float), for offline replay of captured audio
// =============================================================================

extension Pcm16Bytes on Uint8List {
  /// Little-endian PCM16 → float in approx [-1, 1] (app uses /32767 or /32768
  /// in different places; isolate path receives doubles already from the
  /// stream listener which uses /32767.0 for playback and toDoubleList /32768).
  List<double> toDoubleList({double scale = 32768.0}) {
    final byteData = buffer.asByteData(offsetInBytes, lengthInBytes);
    final samples = <double>[];
    for (var i = 0; i < byteData.lengthInBytes; i += 2) {
      samples.add(byteData.getInt16(i, Endian.little) / scale);
    }
    return samples;
  }
}
