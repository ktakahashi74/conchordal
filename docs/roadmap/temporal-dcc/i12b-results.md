# I12b 二経路比較の結果記録

**2026-09-24 21:11 PDT更新：attempt-002は192/192件を統制条件下で取得したが、凍結済み判定は不合格である。** 新rootへの全件取得は終了値0、全192行がcompleted/exit=0、active markerなし、control sessionはcontrolledである。登録後のsource 512件と入力・封印資料のSHAに不一致はなかった。元の中断114件は混ぜていない。

[判定結果](../../../target/handoff-20260924/i12b-attempt-002/evaluation.json)は終了値1、`passed=false`である。`None`の12条件、旧新版fullのWAV6条件、report有無のWAV18組、packet対応18腕、性能非退行120欄、適格性144欄は通過した。一方、旧新版fullのreport回帰6条件と、3音色それぞれの前向き・後向き因果正例が通らなかった。音声の一致だけでreport差や因果鎖の失敗を取り消さない。下記の未取得・中断状態は履歴である。

[attempt結果記録](../../../target/handoff-20260924/i12b-attempt-002/attempt-result.json)に登録・freeze・台帳・control・判定のSHA連鎖を保存した。[独立Astra結果監査](../../../target/handoff-20260924/i12b-attempt-002/results-review.md)は終了値0で完了し、技術完了は不可、失敗終結は可能とした。raw 678点、新電源監査192点、封印source 512件のSHAに不一致はなく、旧114件の保全も確認した。競合build検出は保存されているが、別の時間計測まで不在だったと証明する記録はない。[終結の独立レビュー](../../../target/handoff-20260924/i12b-attempt-002/closure-review.md)も「失敗終結および独立作業への移行：可」とした。[終結記録](../../../target/handoff-20260924/i12b-attempt-002/closure.json)でrawと不合格結果を閉じ、1159件のsource archive（うち封印512件）を比較用基準版として固定した。I12bの技術完了を宣言せず、自動の第3回取得、除外規則や閾値の変更は行わない。個別に登録・レビュー済みの後続実装は、mainの封印sourceを維持したまま、別worktree・別CARGO_TARGET_DIRで開始できる。


**2026-09-24 19:28 PDT更新：取得中断、114/192件完了、技術完了は未判定。** 実取得環境のAC給電を確認して取得を始めたが、115件目の起動前検査でBattery Powerを検出し、runnerが終了値1で停止した。offline48件と性能66件が終了値0で完了し、115件目のchildは未起動である。元の台帳・raw・active markerを保持し、失敗control sessionを修正して再開しない。全体evaluatorは未完の台帳を拒否した（終了値1）。後の読み取りではAC Powerへ戻っているが、中断時の統制失敗を取り消すものではない。[中断と再試行案](../../../target/handoff-20260924/phase0/i12b-interruption-plan.md)、[取得ログ](../../../target/handoff-20260924/phase0/i12b-acquisition.log)、[統制記録](../../../target/i12b-20260924/acquisition/control/session-001/status.json)を参照。独立結果監査と終結記録は未了である。以下の「未取得」「0件」は、この更新より前の準備記録として保持する。

**状態：実装・Rust検証済み、未取得、I12b技術完了未判定。** [実装前登録](i12b-coupling.md)と[入力・判定規則の凍結](i12b-inputs/freeze.json)を保持する。I12bは到来期待がVoiceのonset候補時刻へ作用する前向き経路と、実音の観測accentが周期推定へ入る後向き経路を、それぞれ独立に切って比較する。I5の共有gesture推論に新しい時刻consumerは追加していない。

Rust実装では、前向き切断腕もCDFの生成・配送・適格性を維持し、`arrival_weight = 0`だけを適用する。後向き切断腕は周期pair／peakへのaccent寄与だけを抑え、実accentの台帳受領、到来時計・reset、auditory timingへの配送を保つ。既定`None`と通常`full`では新しい設定値・計数欄をreportへ出さず、従来の形を維持する。[独立コードレビュー](../../../target/i12b-20260924/code-review.md)は封印済みI11-2のsourceを基準に変更11ファイルを確認し、blocking defectなしとした。これは実走の因果効果を示す結果ではない。

[Rust検査記録](../../../target/i12b-20260924/validation-round2/cargo-test-status.txt)は1180件通過、失敗0、無視37。`cargo fmt --all --check`と`cargo clippy -- -D warnings`も終了値0（[fmt](../../../target/i12b-20260924/validation-round2/cargo-fmt-status.txt)、[clippy](../../../target/i12b-20260924/validation-round2/cargo-clippy-status.txt)）。[取得runner最終レビュー](../../../target/i12b-20260924/runner-review-final.md)は静的確認済み。[release事前build](../../../target/i12b-20260924/release-prebuild/status.json)は成功し、sourceの前後SHA一致を記録した。候補binaryの最終封印・取得とは別の記録である。

[検査器レビュー](../../../target/i12b-20260924/verifier-review.md)と[取得前裁定](../../../target/i12b-20260924/verifier-adjudication.md)を反映した独立evaluatorは、[最終レビュー](../../../target/i12b-20260924/verifier-review-final.md)を通過した。現版の検査は[22件成功](../../../target/i12b-20260924/validation-evaluator-round4/status.json)。取得器は、模擬192行を実判定器へ渡す検査を含め[10件成功](../../../target/i12b-20260924/validation-python-round2/status.json)し、[追加検査のレビュー](../../../target/i12b-20260924/runner-test-review-final.md)も通過した。これは実192 runの取得ではない。

観測reportは0.1秒ごとに間引かれるため、すべての決定packetに観測recordがあるとは主張しない。実受領packetの根拠は封印sourceの配送・記録経路と決定recordに置き、対応する観測recordがある場合は内容の完全一致も検査する。最初の周期差は、両busが最後に同値だったcutと最初の差の証拠があるcutの間として報告し、区間内の時刻順を確定したとは言わない。保存済みI11-2のschemaとsource連鎖の確認は[packet監査](../../../target/i12b-20260924/packet-record-source-audit.md)に分けた。

[候補manifest](../../../target/i12b-20260924/candidate/manifest.json)と[実行前seal](../../../target/i12b-20260924/candidate-seal.json)を作成し、source 512ファイル、検証関連238ファイル、検証資料27点、instrumentとrendererのbinaryを固定した。sealのSHA-256は`8a5fe0dad53c7b85a9521e2055620a1c16dbce55164bfb8708c161abbe7247f5`。取得器による準備照合は終了値0で、実行は開始していない。Rust検査には保存済みの完全なround2ログと同一shellの終了statusを使う。後から共有ルートの`test_report.txt`に別の出力が現れたため、そのファイルを上書きせず、[採用記録](../../../target/i12b-20260924/validation-selection-note.json)を残した。候補の全Rust sourceは検証・事前build時のSHAと一致する。

[Astraの最終integrity監査](../../../target/i12b-20260924/final-integrity-review.md)も通過した。source・binary・検査器・検証資料のSHA、および取得準備provenanceを照合し、取得0件を確認した。これは実測の合格を意味しない。

登録192 runの取得数は**0**。三腕の因果正例、`None`／period-on `full`の回帰、120欄の性能非退行はすべて未判定。2026-09-24 17:02 UTCの[測定環境事前検査](../../../target/i12b-20260924/control-preflight/power-states.jsonl)はBattery Powerで、AC給電確認待ち。蓋開放、当該Pythonへのsleep抑止、競合buildなしは確認した。通常sandboxのAC表示と差があるため、取得側の実環境確認を優先する。I11-2の限定受入と原性能不合格は変更せず、R2、A1／A2、A3も別判定のまま。I12aは研究拡張の範囲に残り、このI12bの完了条件へ含めない。
