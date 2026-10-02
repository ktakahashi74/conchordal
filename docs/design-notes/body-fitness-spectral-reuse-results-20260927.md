# 代表身体スペクトル再利用の独立数値比較

日付: 2026-09-27。取得前登録: [body-fitness-spectral-reuse-registration-20260927.md](body-fitness-spectral-reuse-registration-20260927.md)。封印 v13b の `src`、Cargo、fixture を `target/body-fitness-spectral-compare-20260927/v13b-isolated/` にコピーし、比較用 ignored test だけをそのコピーへ追加した。封印本体は変更していない。`diff -qr` でコピーと封印本体の差は `src/life/body_footprint.rs` の比較試験だけ。v13b と v14 は別の Cargo target でコンパイルした。

48 kHz、512 sample/hop、NSGT nfft 16,384・Right alignment、Log2Space 55–8,000 Hz・96 bin/octave、coherent power、tau 10 ms、loudness exponent 0.23、reference power 1e-4。Sine/Harmonic/Modal × brightness 0.30/0.85 × 候補 220/330 Hz の12条件。各条件72 frame の PCM f32 bit、各 frame の主観強度 scan と loudness mass、代表関数の平均 scan と in-band mass を記録した。ADSR は attack 0.005秒、decay 0、sustain 1、release 0.2秒。比較 test は v14 source に保持した。

両版の test は各1件成功。全864 frame、12平均行、12条件行の raw は byte一致。PCMも byte一致。raw SHA-256 は両版 `f37da6af6696c0ccea961303bd7b51135792f11cd9f972218d43e3aa98e4000d`、PCM SHA-256 は両版 `57d9e4e471ca3974e78fcec8b7da23d68f333e9e14b86e4568acc37e46a57c35`。raw とログは `target/body-fitness-spectral-compare-20260927/v13b-48k.raw`、`v14-48k.raw`、各 `.pcm`、`v13b-48k.log`、`v14-48k.log` に保存した。最初の8 kHz試行と同一 Cargo target を使った v14 試行は、実取得条件と版間分離を満たさず最終証拠には用いていない。

既存 focused 検査は peak extraction 10件、body footprint 9件、spectral frontend 2件成功。新しい owned/borrowed と full/部分 reset の比較は Sine/Harmonic/Modal、brightness 2値、候補2値、72 frame で scan、mass、平均、BodyFitness の bit 一致。代表関数の部分 reset は `last_landscape` を更新しない。呼出箇所を監査し、代表結果後の `AnalysisStream::last()` は固定の `space` だけを使う契約を明記した。`process_subjective_frame` の所有値 API は試験専用となり、`cfg(test)` に限定した。

数値一致は処理時間短縮や v13b の frame 432 回復を証明しない。速度と paced 判定は v14 専有取得で別に確認する。
