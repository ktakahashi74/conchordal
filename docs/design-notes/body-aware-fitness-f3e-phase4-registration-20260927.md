# F3e 第四段階: 連続 gate の準備中延期と実数値消費の取得前登録

日付: 2026-09-27。状態: 取得前登録。第三段階と既存 capsule は変更しない。この段階は試験専用の明示配送政策を調べる。通常 runtime への配線、実 thread の費用、有効評価率、公開既定の採用は判定しない。

## 原因と政策

従来の毎 gate 予約は、実行中要求の後ろに常に新要求を置く。完了時に待機要求があれば完成密度を捨てるため、完了遅延が gate 周期より長いと Ready 機会が 0 になる。完成密度を保持するだけでも解決しない。旧 fallback が RNG を進め、非格子 Gaussian 候補を変えるため、次 gate の全候補密度は揃わない。phase3 の連続 fallback 反例と数値回復条件を対照として維持する。

試す任意政策は、身体評価要求の Running 中は proposal gate を延期し、target と RNG を保持する。延期した gate の判断を後から実行しない。完成後の**新しい** gate でのみ、当該 Ready の密度をその時点で受理した F3b source 除去環境と現在の共有 habituation 版で採点する。実 Voice gate が従来の source、birth、body generation、Recipe、route、epoch、current pitch、target、RNG、space、control、habituation の失効判定を行う。準備時の RNG を Voice へ書き戻さない。結果が拒否されたらその gate の旧 proposal に戻し、次要求は commit 後の live 状態から予約する。

延期中の身体・発声・環境・source 状態を凍結する政策ではない。延期中に密度の物理入力や source identity が変わった場合、完成物を流用しない。今回の正例は固定 Sine 身体、固定 pitch、固定 routing、固定 epoch に限定する。変化を許した一般政策の完成とは判定しない。

## 取得条件と成功条件

- 48 kHz、512 sample/hop、1 source、gate 周期 1 hop、仮想完了遅延 8 hop。hop 0 に要求を予約し、hop 0–7 の proposal を延期する。hop 8（sample 4,608）で完了を先に処理し、hop 8 の新 gate に実密度表を渡す。全 hop で `dt=512/48000` を `Voice` の decide と commit に渡し、articulation／body／lifecycle と PitchController の時計・theta位相を通常通り進める。延期するのは proposal 本体のみ。`accumulated_time` は各 hop で加算し、延期中は減算せず、次の実 gate に全経過時間を一度だけ渡す。延期解除の実 proposal 後は残量を 0 にし、次 hop から通常周期へ戻す。過去の gate を後続 adaptation へ二重計上しない。salience と adaptation は延期中に更新せず、その時点の値を保持する。各 hop の音声は固定 Sine 440 Hz からレンダリングする。F3b の fresh batch は hop 8 の判断時点に受理し、共有 habituation も同じ 9 hop の混合 PCM で進める。
- 完了前の gate で proposal RNG、target、salience、adaptation は不変。延期件数 8、Ready 機会 1、実身体表消費 1、実 score 利用 1 以上を別々に記録する。延期中に数値 q を消費したと数えない。
- hop 8 の結果は、同じ現在 RNG・target・環境を使う直接表対照と target、salience、adaptation、終了 RNG、commit 後基音で一致する。旧 habituation 版を束縛した表、変更した body generation の表は拒否され、対応する旧 fallback 対照と一致する。
- 従来の毎 gate fallback と同じ P=1、L=8 の既存反例は Ready 機会 0 のまま。局所固定候補だけを再利用しても、新 RNG の非格子候補が欠ける条件を明示する。
- 結果行は `ready_opportunity`、`numeric_skipped`、`actual_body_consume`、`deferred_gates`、拒否理由を独立して出す。bounded queue や仮想 Ready だけを成功としない。

`cargo test -- --nocapture` の全 stdout/stderr と同一 shell の終了値を `test_report.txt`／`test_status.txt` に記録し、`cargo fmt --all --check` と標準 `cargo clippy -- -D warnings` を実行する。実時間測定は他の cargo/render を停止した専有枠で別途行う。
