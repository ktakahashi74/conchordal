# 身体評価を移動へ接続する前の境界確認

日付: 2026-09-26。状態: F1/F2からF3への技術引継ぎ。新runtime方式の登録・実装・採用ではない。
関連: [全体計画](body-aware-fitness-plan.md)、[数値参照](body-aware-fitness-reference.md)、[最小結果](body-aware-fitness-results-20260926.md)。

この文書は2026-09-26時点のF1/F2→F3接続検討を保存した技術引継ぎであり、旧参照方式を本番へ実装する現行指示ではない。2026-09-29の[再出発記録](body-fitness-runtime-restart-20260929.md)と[主計画の現行工程](body-aware-fitness-plan.md#現行工程2026-10-02)を優先する。特に全候補×72 frameの実合成・NSGTは数値参照とし、本番用身体表現の契約を別に定義する。

## 接続箇所と二つの計算段階

`PitchHillClimbPitchCore::propose_with_scorer` は候補だけを採点する関数ではない。
同じscorerで局所／大域の格子を調べ、候補を抽出してから、非格子上の近傍・乱数候補と現在位置も採点する。
したがって、最後の候補のscoreだけを身体評価へ差し替えても、候補抽出が旧一点評価に依存したままになる。
二方式の比較では、候補抽出も含めて変える比較と、あらかじめ固定した共通候補だけを再採点する比較を区別する。
既存コードの既定は大域peak候補0、ratio候補無効だが、局所格子と3個の乱数候補は使う。

F3の接続には、少なくとも次の二段階の準備が要る。

1. 身体recipeと候補基音から作る代表密度。body generationと代表条件の同一性を持ち、基音ごとの絶対周波数解析を使う。
2. sourceを除いた同じ解析区間の環境と、その時点の共有habituationで各候補を採点する処理。

身体密度の計算は環境とは独立なので再利用できる。一方、現在の参照関数は候補ごとにToneを再合成し、
解析器をresetして観測区間全体を処理する。これを音声callbackや既存同期scorerの一呼出しへ直接入れる設計は採らない。
補正済み密度scanの単なる横移動、任意基音の格子への丸め、格子間補間を正解参照と同一視しない。
そうした近似を使うなら、C誤差だけでなく候補順位・採否の差をF2で別途登録して検証する。

## 解析履歴の開始条件

出生前の共有AnalysisStreamを複製すれば、その新sourceを含まない過去から自己除去を始められる。
途中から既存sourceの自己除去を開始する場合、そのsourceを含んだ共有解析履歴を複製するだけでは足りない。
NSGTの窓と前処理の平滑化に自己音が残るためである。有限の無音warmupを厳密な履歴再構成として扱わない。
最初の小集団版では起動時または出生時から登録する範囲を固定し、途中登録の可否を明示する。
欠落したPCM区間をゼロとして埋めた結果も、有効な自己除去へ数えない。

退役したsourceのtailは新生sourceの環境に残す。追跡対象を切り替える際はsource id／世代を照合し、
現在のVoice一個の基音だけで、残っている複数Toneの所属を推測しない。route変更とbody generation更新も別の事象である。

## 配送と比較で固定するもの

workerから返る計算結果にはsource・body generation、recipe、候補集合、解析設定epoch、支持終端、
利用可能時刻を対応づける。一判断で異なる環境区間の結果を混ぜない。habituationも同じ時点の共有履歴を使う。
計算の完了を待った後で、その結果を過去のgateや出生時刻へ適用しない。

候補集合を先に作るか、格子評価の後に作るかによって、乱数を消費する時点と意思決定時点が変わる。
非同期jobへ共有乱数器を渡したり、古いコピーの乱数状態を後から書き戻したりしない。
方式の比較では共通の乱数入力・候補・計算期限を確保し、配送時刻の差による行動差を評価式の効果に混ぜない。
結果の欠測・期限切れ・世代違いを理由付きで報告し、旧評価へ戻った時間を身体評価の有効利用へ含めない。

まず1／4 Voiceの固定身体で出力の鮮度と処理費用を測る。64 Voiceを同じ方式で常時解析できるとの前提は置かない。
worker・queue・保持量の上限、期限と有効利用率は実装前の登録へ落とし込み、I4第二段階の配送境界と照合する。
現在の最小参照試験の合格や、並行ジョブ下のテスト所要時間を実時間資源の合格として引き継がない。

## 実Voiceの一判断へ進む際の接続点

F3cは実Voiceが出した音を観測する試験であり、pitch decisionを身体評価に変更しない。
次の接続点は `Voice::decide_pitch_target_with_listener_pressure` から呼ばれる
`PitchController::update_pitch_target` の `should_propose` 分岐である。
`force_set_target_pitch_log2` で結果だけを注入するとgateと適応更新を迂回するため、比較の入口にしない。

単一の `ReceivedBatch::accept` で受理したsource別Landscapeに、判断時点の共有habituationを一度適用する。
その同じ有効Cと事前準備済みの各候補密度からF2のscore表を作り、float bitで候補基音を照合する。
`propose_with_scorer` の局所候補抽出と最終採点の双方へ、同じ表を返す。
`propose_target_with_crowding_salience` は旧Cと旧ExactScanを内蔵しているので、そのまま流用しない。
既存のobjective、landscape weight、移動・tessitura・crowding・適応の補正順序は維持し、
`adjusted_pitch_score_impl` が読む基礎Cだけを置き換える境界を設ける。
proposal後のtarget・salience・適応更新、PitchModeのrange clamp、Voiceのcommitも維持する。

準備用RNGの複製と実PitchControllerのRNGを照合するだけでは、非同期の予約方式は完成しない。
最初の一判断試験では準備から消費までRNG・target・pitch制御・身体recipeが変わらない条件を固定する。
一般の非同期接続では、その条件が崩れたrequestを破棄する手順を実装する。
表にない候補や密度質量ゼロをscorer内で検出して途中まで乱数を消費するのでなく、
判断に入る前に全入力の対応を検査する。古いRNG状態の書き戻しは行わない。

F3bの `SourceIdentity` はsource id・Voice generation・birth sampleを持つが、身体世代は持たない。
候補側の `footprint::Identity` とcapture tokenのbody generationは別に照合する。
`Voice::footprint_recipe(fs)` の周波数は現在基音なので、候補ごとの生成時にはcandidateの周波数を使う。
この境界に基づくF3dの固定条件・一判断接続は実装・検査を通過した。
実Voiceの440 Hzから466.163635 Hzへの移動、commit、旧Cに依存しないことを確認した。
一般の状態変更を含む非同期接続、実時間の鮮度・有効評価率は未検証である。
