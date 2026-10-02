# F4e PeakBiased 出生一機会の事前登録（2026-09-27）

## 範囲と境界

F4b/F4c のオフライン Entry と照合 context を再利用し、`RespawnPolicy::PeakBiased` の一回の死亡・補充機会を検査する。候補の身体は、実際の子に使う population template、予約された子 ID・世代 0、候補周波数、同じ landscape と seed から `spawn_with_landscape` で確定する。身体ごとの代表的な音響 footprint から fitness score/level を算出し、抽出済み候補の選択重みと補充閾値へ渡す。選ばれた候補と実際に生成された子の身体・周波数・recipe identity を照合する。

対象構成は parent のいない PeakBiased、一つの死亡と一つの補充枠。range 内の**全 Log2 bin 中心**で身体を評価し、その body score から最大 16 個の peak 候補を抽出する。候補 bin ごとに非ゼロ半径の局所探索 grid も事前列挙・身体評価し、選択された bin の局所探索結果と実際の子を照合する。局所探索の候補ごとに周波数依存の身体が変わり得るため、bin 中心の recipe をずらして使い回さない。初回 Spawn の Field 配置、parent ありの PeakBiased、通常 runtime への observer 配線は対象外。接続は `cfg(test)` のオフライン境界。

## 取得前に固定する判定

1. 共有 habitat の実解析 C scan と、全 range bin・最大 16 peak の局所探索点ごとの実身体 footprint を使う。footprint は各 72-hop。費用は全 bin 数と探索点数に比例する。壁時計性能は測らない。
2. 抽出済み peak の選択について、同じ seed から parent 不在時の fallback 配置による RNG 消費を再現した後、body score の非負値を `scene_score_exponent` 乗し `WeightedIndex` で選ぶ独立参照を置く。局所探索の独立参照は、中心と同じ grid を列挙し body score の最大を採る。選択後の RNG probe を比較する。点 score scan を変更しても body bin 抽出・選択が同じになる対照を設ける。
3. 低閾値で一人だけ生成、高閾値で生成ゼロ。低・高の点 level を与えても、同じ body score なら閾値判定は同じ。観測 counter は機会あたり一増、拒否時の子 ID・member index は未消費。同じ死者の次 hop に重複補充なし。
4. context は population、member index、予約子 ID、source IDs、frame、解析 epoch、支持時刻、候補順・周波数・body・recipe identity を照合する。stale epoch、別子 ID、別 template、候補周波数の不一致は選択前に拒否し、子と event を生成しない。
5. 結果は実装前登録との差分を明記して保存する。対象テスト、全 `cargo test -- --nocapture`、`cargo fmt --all`、標準 `cargo clippy -- -D warnings` を実行する。`test_report.txt` と `test_status.txt` は同一 shell で記録する。
