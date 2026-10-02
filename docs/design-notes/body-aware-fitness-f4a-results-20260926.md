# F4a: 成人一更新の隔離検査結果

2026-09-26。判定: 取得前登録の一更新検査は成立した。通常runtimeへの身体代謝採用やF4全体の完了を意味しない。

取得前の[登録](body-aware-fitness-f4a-metabolism-registration.md)をSHA-256 `4d7378282ad059273c4d08255f5244d09bff3bbb36f99e4df25c74c7ed29f764`で固定した。実装は基準commit `06a4772c43d06b41b44753bb0be891f23e16b93e`に封印F3dの19 source（tar SHA-256 `ca5c0f7cc813793a75b97c7c210bd7d17574720c8fecb37598c3c71563790011`）を重ねた隔離worktree `.worktrees/body-fitness-lifecycle`。F3dの19 sourceを再照合すると、18件は封印hashと一致し、`src/runtime/mod.rs`だけがF4a試験moduleの宣言で変わった。mainの`src/`や既定設定は変更していない。

`Voice::commit_decided_control`の既存順序に沿い、autonomous更新の直後に`cfg(test)`のスレッド限定contextで実Voiceの現在基音とBodySnapshotを読む。64 hopの実混合／source別PCMをF3b workerで処理し、source 1自身を除いた受理環境と、72 hopの実Toneから得たqでF2のscore/levelを一度計算する。lifecycleの基礎費用・連続recharge・内部attack、`LifeAccumulator`、Gated onsetのrechargeに同じ身体評価を配送する。phonation gateの判断には従来の点Cを残す。受理source・epoch・支持終端・受信時計・使用時の現在基音が不一致なら試験を失敗で止める。Gated onsetのbindingと鮮度はengine tickより前に検査する。

seed 7、48 kHz、512 sample/hop、Harmonic 440 HzとSine 466 Hz、Glide target 466 Hz・時定数0.02秒・制御dt0.01秒の固定fixture。autonomous後の実基音は450.052521 Hz、身体scoreは−0.182956919、levelは0.409528732、in-band massは25.988426。target基音でのscoreとの差が`1e-5`を超えることを試験内で確認した。休符一更新のenergyは、旧点Cの低値／高値に対し身体入力で両方0.403300047、context無しの旧入力では0.399524987／0.408975005。閾値窓`[S−0.3,S−0.1]`と`[S+0.1,S+0.3]`では0.409204751／0.399204761となり、既存代謝式のscore信号1／0と合う。内部attackは1件、energy 0.435205787で基礎費用＋連続recharge＋一回のattack式と一致。Immediate gateの実Gated onsetは点C低値／高値の双方でsample 32768にstrength 1の一件、発声command列も一致し、rechargeだけ身体levelを使用した。WhenViable gateは点C低値でonset 0、高値で1件となる。source id／出生世代／epoch／期限／現在基音／登録身体世代の不一致を拒否し、期限切れonsetではclock未実行、既存onset出力・energy不変を確認した。

最終`cargo test -- --nocapture`はexit 0（lib 1159件中1119 pass・40 ignored、他の統合試験も失敗0）。`cargo fmt --all --check`、通常`cargo clippy -- -D warnings`も成功。追加の`cargo clippy --tests -- -D warnings`はF4a外の既存試験箇所にある17警告で失敗したため、試験込みClippy合格とは記さない。最終全testの前に、既存の一時directoryとの名前衝突で統合試験1件が`EEXIST`となった回を失敗として保存した。専用の新規`TMPDIR`で同じsourceの全testを再走して成功した。

取得前登録が求めた入力記録は[最終補足capsule](../../target/body-fitness-f4a-supplement-20260926/offline-evidence/f4a-result.json)に保存した。64 hopの混合・自己・他者habitat PCM、72 hopの現在基音実Tone PCMをf32 little-endianで保持。受理sourceのepoch・世代・支持終端・受信時刻、Log2Space、実効C scanとsource除去後の密度、代表q密度・recipe identityを同階層に保存した。保存したTone PCMを別の解析器で処理したq密度が、代謝入力の計算に使った密度とf32 bit単位で一致することも確認した。詳細JSONには各群のbody／identity、energyの既存式による期待値と実値、`LifeAccumulator`の前後、二種類のattack/onset件数、拒否理由、発声列を記録した。元の[初版capsule](../../target/body-fitness-f4a-capsule-20260926/result.json)は改変せず、最終sourceと検証結果は補足版の全source tar・225件の個別hash・29件のartifact hashで固定した。

基準版との旧`None`直接対照は、入力を実行前に固定したseed 7・固定Harmonic 440 Hz・Entrain＋Gatedの3秒fixtureで、封印済み06a release renderとF4a隔離release renderを順次実行した。[比較記録](../../target/body-fitness-f4a-supplement-20260926/base-none-regression/comparison.json)のWAVはbyte一致（SHA-256 `0e12b2c8dd859a2c8451dbb6695b5cfe3aa596168b056b094456be0f8b78849e`）、RMS 0.003250で無音ではない。onsetは両者ともsample 16847の一件、deathは一件で`energy_depletion_sec=0.54666543`、`first_k_mean=0.1`。`hop_timing`301件を除く全1586件のreport recordは構造一致した。生のWAV・JSONLとbinary／入力hashも保存した。`hop_timing`の実時間値は一致判定の対象外。

このVoiceには出生世代とは独立した身体世代fieldがない。上の身体世代0は固定fixtureのrecipe識別子であり、一般の身体変更を検出する証明ではない。通常runtimeでの身体評価配送、期限切れ時の本番fallback、出生・親選択、連続生存、正式資源受入も未検査。基準版との一致は上記3秒の固定fixtureの範囲であり、一般の入力全体に対するbit同値証明ではない。
