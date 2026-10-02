# F3 明示drive付き自己除去処理器の追加費用probe

日付: 2026-09-26。状態: 実装・取得前登録。先行する [参照費用probe](body-aware-fitness-cost-probe-registration.md) の14条件、source／binary capsule、結果を維持し、別の `#[test] #[ignore]` release取得として追加する。先行4 source条件ではtimed区間の最小own PCM RMSが約6.30×10⁻⁹であり、holdとSeqGateを延長しても全sourceの持続励振を示せなかった。この追加条件はその測定値を上書きしない。

`build_analysis_runtime_core(&AppConfig::default(), 48000)` を直接使用する。1 sourceと4 sourceの二条件。Sine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hzの順で、振幅0.35、brightness 0.8、onset kick 1、hold 480000 sample、SeqGate 10秒、ADSR attack 0.005秒・release 0.5秒を固定する。各Toneの出生batchに `ToneCmd::Update` を加え、at_tick=0、`continuous_drive=Some(1.0)`、他のupdate欄はNoneとする。Sineはbackend仕様上driveを消費しないが、同じ入力commandを明示する。

混合PCMとsource別PCMを独立 `ScheduleRenderer` で320 hop事前合成し、合成とrenderer生成は計時外。64 hop warmup後の256 hopすべてで、混合PCMのRMS > 0、各own PCMのRMS ≥ 1e-4を必須とし、各sourceの最小RMSと全体最小RMSを記録する。閾値未達なら失敗とし、取得後にdriveや閾値を調整して成功条件へ入れない。1 sourceの自己除去Landscape massは0、4 sourceの各自己除去Landscape massは正を必須とする。主要scanは有限・Log2Space整列、identityと支持終端は正しく照合する。

各反復で新規 `Processor` を計時外に構築し、frame 0出生の`step`を別計時、64 hopのwarmup、続く256 hopの各`step`を計時する。三反復、定常標本768件／条件と出生3件／条件。計時にはshared一回、source一／四個の自己除去・解析・C再計算・出力構築を含み、出力検証と破棄は含まない。`Instant`、`black_box`、全標本、中央値・nearest-rank p95・最大、10.6667 msのhop予算比をJSONに記録する。正式資源合否閾値は置かない。

数値参照試験と区別して取得する。取得前にsource・binary・この登録のSHA、コマンドとホスト競合状態を保存し、専有窓で `--ignored --nocapture --test-threads=1` の対象testだけを実行する。全stdout／stderr、終了コードと検査結果を新しい出力ディレクトリへ保存する。先行probeや旧F3a capsuleへ書き込まない。この値は解析処理器の予備費用で、音声callback、queue、結果配送、通常runtime総hop、16／64 source展開を含まない。
