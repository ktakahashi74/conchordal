# 退役時に短い記憶区間を保存する実験

日付: 2026-09-27。状態: 試験専用の探索結果。通常モデル・既定設定には未採用。
関連: [統合検証](body-fitness-integration-validation-20260926.md)。

## 問題と介入

両ONのModal試験では、追跡容量による群の交代が96 hopの保存窓より早く起き、
記憶episodeが一度も保存されなかった。今回の介入は、退役時に未完成の区間を保存することだけである。
時刻と欠測区間、奇数hop終端をそのまま保持し、退役した時刻以降に利用可能とする。
支持量の分母は元の96 hopに固定し、短い断片を長い区間と同じ支持量へ拡大しない。
比較には既存の二つの完成bin以上・coverage 0.9を使い、bank 32・group 7・公開参照16の上限を維持した。

Sine／Harmonic／Modal、seed 20260918／17／29、機能OFF／ONの18本を取得した。
取得前planのSHAは `1a3bb3de6fa451da7f91cb86af6c0eb05f6044628a90a81bca0822040a719f3d`。
sourceと結果は `.worktrees/integration-memory-diagnostic/target/memory-partial-window-20260926/` に保存した。
最終manifestのSHAは `4c9ffba754a08f6f793c9f5e86acabb1ce90a701f171a566f3e6c79d54d21c5f`。

## 保存と参照の結果

18本とも終了し、容量、保存会計、照合時刻の因果順序の検査を通過した。
以下はModalの各busあたりの件数である。二つのbusを合算した件数ではない。

| seed | OFFの保存総数 | ONの部分区間保存 | ONの保存総数 | OFF→ONの参照在庫が正のreport frame数 |
|---|---:|---:|---:|---:|
| 20260918 | 0 | 200 | 200 | 0→8 |
| 17 | 1 | 180 | 181 | 0→17 |
| 29 | 0 | 171 | 171 | 0→16 |

Harmonicでも147／175／152件の部分区間が保存された。Sineでは全seedで退役時保存の試行が0だった。
全9組でON/OFFのWAV byteと参加判断は一致した。記憶参照が増えたことを音楽的改善と解釈しない。

## 独立した事後診断: 学習経路への到達

保存・参照在庫と学習を区別するため、取得済みの全18 reportの`private_participation_trace`を
別scriptで集計した。追加取得や閾値変更はしていない。Modalの記録はhabitat bus 0のonset／releaseに属する。

| seed | ONの非null参照entry総数 | 参照を持ちapplied=trueのtrace数 | 正のcreditを持つentry数 | prior以外の予測を持つentry数 |
|---|---:|---:|---:|---:|
| 20260918 | 49 | 6 | 2 | 0 |
| 17 | 498 | 28 | 270 | 9 |
| 29 | 193 | 8 | 90 | 13 |

OFFではこの表の値はすべて0だった。ONでは参照が学習入力へ届き、正のcreditも記録された。
ただしseed 20260918では参照予測がすべてpriorに留まる。entryの件数は独立試行数ではなく、
統計的な優越や長期学習効果を示すものではない。音声・参加判断の一致とも両立する。
script、全入力SHA、結果は同worktreeの `target/memory-partial-window-trace-diagnosis-20260927/` に保存した。

## 不合格と保証の境界

取得前登録には、既存seedのOFFで旧phase2の全report fieldが一致するという条件があった。
これは3素材とも不成立だった。WAVは一致したが、報告行数、非同期body workerの候補配送時刻・集合等に差がある。
したがって主比較の `validation_pass=false` を保持する。事後の補助診断では、phase2で既に登録されていた
限定範囲（10種の学習記録、非診断temporal、参加判断、CDF、共通候補値、WAV）は3素材とも一致した。
この補助比較で、取得前の全field条件を差し替えない。

比較scriptには表示式の括弧の誤記もあった。原版と2回の構文失敗を保存し、表示式だけ修正して集計した。
全cargo testは1231成功・0失敗・43 ignore、formatと標準Clippyは通過した。
元の統合25条件の全体不合格、通常モデル未採用、R2未判定は変わらない。
