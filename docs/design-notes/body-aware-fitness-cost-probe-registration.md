# F3 接続前の参照計算費用プローブ

日付: 2026-09-26。状態: 取得前登録。対象は隔離 worktree の `#[test] #[ignore]` で実行する release 計測だけで、通常 runtime の合否判定ではない。実行前に source と binary の識別、計測機の状態、コマンド、出力先を保存する。数値参照テストとは分離する。

## 共通条件と記録

`build_analysis_runtime_core(&AppConfig::default(), 48000)` から解析設定を直接得る。サンプルレートは 48 kHz、既定 hop は 512 sample、1 hop の時間は 10.6667 ms。`std::time::Instant` で計時し、結果を `black_box` に渡す。各標本をマイクロ秒単位で記録し、昇順の中央値、nearest-rank 95 パーセンタイル、最大値と hop 時間に対する比率を JSON 一行で stdout へ出す。サンプル数、条件、有限性検査、測定方法も同じ JSON に入れる。RSS の増減はこのプローブのメモリ予算ではない。

## 代表身体密度

`representative_subjective_intensity` の既存実装を変更せず計測する。Sine、Harmonic、Modal の三身体、それぞれ 220／440／880／1760 Hz。source id は 2、amp scale と onset kick は 1、hold は 48000 sample、brightness は 0.8、SeqGate は 1 秒、ADSR attack は 0.005 秒、release は 0.5 秒、その他の身体値は参照試験と同じ既定値。観測窓は 72 hop（0.768 秒）。各身体・候補について初回一回を warmup とし、その後の五回を個別計時する。解析 stream の構築は計時外とし、計測対象は代表関数内の reset、Tone 作成、render、解析、scan 集計を含む一回の呼び出し全体。各出力の観測 frame 数、in-band mass と scan の有限性を検査する。unsupported は成功値へ置き換えず、その条件と理由を記録して失敗させる。

## 自己除去処理器

一 source と四 source を別条件として測る。source は Sine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hz の固定順。各 source の Tone は参照試験と同じ振幅 0.35、brightness 0.8、attack 0.005 秒、release 0.5 秒。ただし processor 条件では hold を 480000 sample、SeqGate を 10 秒に延ばす。混合 PCM と source 別 PCM は独立 `ScheduleRenderer` で少なくとも 320 hop を計時外で事前合成し、入力フレームを保持する。`Processor::new` は計時外。各反復では frame 0 の出生 `step` を別に計時し、最初の 64 hop を warmup、続く 256 hop の `step` を一フレームずつ計時する。これを新規処理器で三反復。計時区間は shared 一回と source 一／四個の解析・自己除去・C 再計算・出力構築を含む。出生時複製の費用は出生 frame の別記録に含む。timed 256 hop の混合 PCM と各 source PCM の RMS > 0 を事前検査する。四 source の各自己除去出力は解析 mass > 0、一 source の自己除去出力は厳密な無音で mass = 0 を検査する。最小 PCM RMS と四 source の最小 mass を JSON に記録する。出力の source 数、identity、support end、mass と主要 scan の有限性も検査する。拒否は理由を記録して失敗させる。

取得前修正: 最初の草案の processor 側 hold 48000 sample と SeqGate 1 秒では、320 hop（約 3.413 秒）の計時区間の大半が発音後になり、持続発音中の解析費用を過小評価する。このため processor 側だけ上記の持続時間へ延長した。また一 source の自己除去は定義上無音になるため、この条件の解析 mass > 0 という誤った草案条件を mass = 0 の対照へ訂正した。代表密度の 72 hop 条件は変更しない。

## 解釈境界

この数値は指定密度・指定音色・指定解析設定での参照処理費用。音声 callback、queue、結果配送、候補選択、共有 runtime の総 hop 費用、CPU 競合下の tail latency、16／64 source 拡張を含まない。事前の正式合否閾値は置かず、10.6667 ms との比率を探索的に示す。F3 接続の可否判断には worker 境界、鮮度、総資源の別測定が必要。
