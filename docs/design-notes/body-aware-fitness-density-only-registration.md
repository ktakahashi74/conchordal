# F3 代表身体密度の前処理専用経路

日付: 2026-09-26。状態: 実装・取得前登録。対象は隔離 worktree の `cfg(test)` 参照実装であり、通常 runtime の身体fitness採用、音声callback、資源合否を含まない。既存の代表身体密度は一候補72 hopで約42～43 msだったが、この観測値から改善量を仮定しない。旧費用probeのbinary・source capsuleと14条件の結果は上書きしない。

## 変更範囲と同一性の理由

`AnalysisStream` に crate 内かつ `cfg(test)` の `process_subjective_frame(&mut self, audio: &[f32]) -> SpectralFrame` を一つ追加する。入力は厳密に一つの完全なhop長とし、長さ不一致をassertする。既存 `process` は変更しない。新メソッドは同じ `RtNsgtKernelLog2::process_hop` の結果を同じ `SpectralFrontEnd::process_nsgt_power` に渡し、`dt_sec = audio.len() as f32 / params.fs` を維持する。R/H計算とLandscapeの結果複製を行わない。`last()`のLandscapeは新メソッドでは更新されず、代表密度処理中には固定Log2Spaceのgeometry参照にだけ使う。この経路を通常のLandscape結果と混用しない。

既存 `AnalysisStream::process` はNSGTと前処理で `subjective_intensity` と `loudness_mass` を決めた後にR/Hを計算する。後段のR/HからNSGT、前処理状態、次frameの密度への帰還はない。旧経路のNSGT出力からLandscapeへのcopyは同じf32 bitの複製なので、同じ入力・状態・パラメータなら前処理専用経路の各frame密度とmassはbit一致するはずである。構築時のroughness参照計算はこの最小変更では残る。密度をPMFに正規化したり、時間平均powerを後から一度だけ前処理したりしない。

`representative_subjective_intensity` のframeループだけ新メソッドへ切り替える。Tone生成、kick、modal位相seed、stream reset、frame順、各frame密度のf32積算、massのf64積算、平均とunsupported判定は維持する。既存 `AnalysisStream::reset` はNSGT、前処理状態、Landscapeを同じように初期化する。新経路でもこのresetを呼ぶ。

## パラメータ境界と数値検査

通常構築は `build_analysis_runtime_core(&AppConfig::default(), 48000)` を直接使う。両streamは同じ `core.lparams` と `core.nsgt` から独立に作る。通常条件のLog2Space、NSGTのnfft／hop／alignment／coherent power mode、RtConfigの帯域別時間定数、SpectralFrontEndのERB格子・peak抽出・A-weighting・基準powerを共有する。前処理の時間刻みは `audio.len() / fs`、既定tauは10 ms、loudness指数は0.23、基準powerは1e-4。NSGT内部の帯域別tau（既定5～20 ms）と前処理tauを混同しない。

独立した旧 `AnalysisStream::process` と新メソッドに同じ実Tone PCMを一frameずつ渡し、毎frameの密度全binとmassを `to_bits()` で比較する。Sine／Harmonic／Modalの各220／440／880／1760 Hz、12条件×72 hopを固定。各条件で旧経路から独立に積算した平均密度・平均massを、変更後の代表関数の出力とbit比較する。新旧をそれぞれresetして同じPCM列を再実行し、全frameと平均が再現するか検査する。無音入力、短いholdの発音後tailも別に全frame比較する。さらにtauだけ25 ms、loudness指数だけ0.31、基準powerだけ2e-4へ動かした条件を別々に検査し、固定値の写し込みを検出する。完全hop以外はassertする境界試験を置く。既存F1/body_footprint試験も回帰対象とする。

## 後続の費用比較

数値同一が通過してから、同一release binary・同一ホスト専有窓で旧full解析と新密度専用解析を交互順に測る別取得を登録する。旧版はこの比較専用のテストhelperで現在の代表関数のTone生成・reset・72 frame積算を写し、frame解析だけ旧 `AnalysisStream::process` を呼ぶ。比較する12条件は上記と同じ。同じrecipeとsource seedから生成するPCM全frameのbit一致を事前確認する。各条件で両群を一回ずつwarmupした後、`旧→新` と `新→旧` の組を各五回、合計各群十回計時する。stream構築は両群とも計時外、Tone生成・reset・render・frame解析・集計を各呼出しの計時内に置く。`Instant` と `black_box` を使い、各標本、ペア差、中央値・p95・最大値を保存する。独立したPCM生成を片群だけ計時外に移さない。ホストのcargo／rustc／render競合不在、binary/source/登録文書SHA、全command/log/exit codeを保存する。旧14条件の別時刻の値との単純差を正式な速度改善としない。数値改善を予想値として登録しない。
